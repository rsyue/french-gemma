"""
Unit and integration tests for Direct Preference Optimization (DPO),
token log-probability extraction, preference dataset collation, and DPO training strategy.
"""

import pytest
import torch
import torch.nn as nn

from src.dataset import train_custom_tokenizer
from src.dpo_dataset import (
    DPODataset,
    format_dpo_pair,
    get_batch_logps,
    get_dpo_dataloader,
)
from train.builder import TrainingFactory
from train.strategies.dpo import DPOStrategy


@pytest.fixture
def mock_tokenizer(tmp_path):
    sample_texts = [
        "<start_of_turn>user\nQuelle est la meilleure ville ?<end_of_turn>\n"
        "<start_of_turn>model\nParis est souvent considérée comme magnifique.<end_of_turn>\n",
        "<start_of_turn>user\nComment coder en Python ?<end_of_turn>\n"
        "<start_of_turn>model\nInstalle Python et commence par les bases.<end_of_turn>\n",
    ]
    tok_dir = str(tmp_path / "tokenizer_dpo")
    return train_custom_tokenizer(sample_texts, vocab_size=250, save_dir=tok_dir)


def test_format_dpo_pair(mock_tokenizer):
    prompt = "Bonjour, qui es-tu ?"
    chosen = "Je suis FrenchGemma, un assistant IA français."
    rejected = "Je ne sais pas."

    pair = format_dpo_pair(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        tokenizer=mock_tokenizer,
        max_seq_len=64,
    )

    assert "chosen_input_ids" in pair
    assert "chosen_labels" in pair
    assert "rejected_input_ids" in pair
    assert "rejected_labels" in pair

    assert len(pair["chosen_input_ids"]) == 64
    assert len(pair["rejected_input_ids"]) == 64

    # Both chosen and rejected should mask the prompt with -100
    prompt_str = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    prompt_len = len(mock_tokenizer.encode(prompt_str, add_special_tokens=False))
    for i in range(prompt_len):
        assert pair["chosen_labels"][i] == -100
        assert pair["rejected_labels"][i] == -100

    # Unmasked labels should match input_ids
    assert pair["chosen_labels"][prompt_len] == pair["chosen_input_ids"][prompt_len]
    assert pair["rejected_labels"][prompt_len] == pair["rejected_input_ids"][prompt_len]


def test_dpo_dataset_and_dataloader(mock_tokenizer):
    pairs = [
        {
            "prompt": "Explique la photosynthèse.",
            "chosen": "La photosynthèse est le processus par lequel les plantes convertissent la lumière en énergie.",
            "rejected": "C'est quand les plantes mangent du soleil.",
        },
        {
            "prompt": "Donne un synonyme de rapide.",
            "chosen": "Véloce ou prompt sont de bons synonymes.",
            "rejected": "Lent.",
        },
    ]

    dataset = DPODataset(pairs=pairs, tokenizer=mock_tokenizer, max_seq_len=64)
    assert len(dataset) == 2

    loader = get_dpo_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert "chosen_input_ids" in batch
    assert "rejected_input_ids" in batch
    assert batch["chosen_input_ids"].shape == (2, 64)
    assert batch["rejected_input_ids"].shape == (2, 64)


def test_get_batch_logps():
    # Batch size 2, seq len 4, vocab size 5
    logits = torch.zeros(2, 4, 5)
    # Give high logit to token 2 at step 1 and token 3 at step 2
    logits[0, 0, 2] = 10.0
    logits[0, 1, 3] = 10.0

    labels = torch.tensor([[-100, 2, 3, -100], [-100, -100, -100, -100]])
    logps = get_batch_logps(logits, labels)

    assert logps.shape == (2,)
    # Sample 0 should have high logprob (close to 0)
    assert logps[0].item() > -0.1
    # Sample 1 (all masked) should have 0.0 logprob
    assert logps[1].item() == 0.0


def test_dpo_strategy_loss_computation():
    class MockModel(nn.Module):
        def __init__(self, high_on_chosen: bool = True):
            super().__init__()
            self.high_on_chosen = high_on_chosen

        def forward(self, input_ids, attention_mask=None, **kwargs):
            bs, seq_len = input_ids.shape
            logits = torch.zeros(bs, seq_len, 20)
            if self.high_on_chosen:
                for i in range(bs):
                    for j in range(seq_len - 1):
                        next_token = input_ids[i, j + 1].item()
                        # Higher probability for chosen sequence (first half of concatenated batch)
                        logits[i, j, next_token] = 10.0 if i < (bs // 2) else 0.0
            class Out:
                pass
            out = Out()
            out.logits = logits
            return out

    policy_model = MockModel(high_on_chosen=True)
    ref_model = MockModel(high_on_chosen=False)

    strategy = DPOStrategy(ref_model=ref_model, beta=0.1)

    # Batch with chosen and rejected
    batch = {
        "chosen_input_ids": torch.tensor([[1, 2, 3], [1, 4, 5]]),
        "chosen_attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        "chosen_labels": torch.tensor([[-100, 2, 3], [-100, 4, 5]]),
        "rejected_input_ids": torch.tensor([[1, 6, 7], [1, 8, 9]]),
        "rejected_attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        "rejected_labels": torch.tensor([[-100, 6, 7], [-100, 8, 9]]),
    }

    loss = strategy.compute_loss(policy_model, batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0

    metrics = strategy.latest_metrics
    assert "reward_accuracy" in metrics
    assert "reward_margin" in metrics
    assert metrics["reward_accuracy"] == 1.0


def test_training_factory_builds_dpo_strategy():
    strategy = TrainingFactory.build_strategy("dpo")
    assert isinstance(strategy, DPOStrategy)


def test_dpo_strategy_label_smoothing():
    class DummyModel(nn.Module):
        def forward(self, input_ids, attention_mask=None, **kwargs):
            bs, seq_len = input_ids.shape
            logits = torch.randn(bs, seq_len, 10)
            class Out:
                pass
            out = Out()
            out.logits = logits
            return out

    model = DummyModel()
    strategy = DPOStrategy(ref_model=None, beta=0.1, label_smoothing=0.1)

    batch = {
        "chosen_input_ids": torch.tensor([[1, 2, 3]]),
        "chosen_labels": torch.tensor([[-100, 2, 3]]),
        "chosen_attention_mask": torch.tensor([[1, 1, 1]]),
        "rejected_input_ids": torch.tensor([[1, 4, 5]]),
        "rejected_labels": torch.tensor([[-100, 4, 5]]),
        "rejected_attention_mask": torch.tensor([[1, 1, 1]]),
    }

    loss = strategy.compute_loss(model, batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0


def test_load_dpo_pairs_file_handling(tmp_path):
    import json

    from train.dpo import load_dpo_pairs

    # Non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_dpo_pairs(str(tmp_path / "non_existent.jsonl"))

    # Valid JSON file
    json_path = tmp_path / "dpo.json"
    data = [
        {
            "prompt": "Test prompt",
            "chosen": "Test chosen",
            "rejected": "Test rejected",
        }
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    loaded = load_dpo_pairs(str(json_path))
    assert len(loaded) == 1
    assert loaded[0]["prompt"] == "Test prompt"


def test_dpo_checkpoint_rotation_logic(tmp_path):
    import os

    output_dir = str(tmp_path / "checkpoints_dpo")
    os.makedirs(output_dir, exist_ok=True)
    best_checkpoints = []
    max_checkpoints = 5

    def mock_save_if_better(step_idx: int, current_loss: float):
        checkpoint_name = f"dpo_checkpoint_step_{step_idx}_loss_{current_loss:.4f}.pt"
        checkpoint_path = os.path.join(output_dir, checkpoint_name)
        should_save = False
        if len(best_checkpoints) < max_checkpoints:
            should_save = True
        else:
            worst = max(best_checkpoints, key=lambda x: x["loss"])
            if current_loss < worst["loss"]:
                should_save = True
                if os.path.exists(worst["path"]):
                    os.remove(worst["path"])
                best_checkpoints.remove(worst)
        if should_save:
            with open(checkpoint_path, "w") as f:
                f.write("mock_dpo_checkpoint")
            best_checkpoints.append({"path": checkpoint_path, "loss": current_loss, "step": step_idx})
        return should_save

    # Save 5 checkpoints
    for step, loss in enumerate([1.5, 1.3, 1.1, 0.9, 0.7], start=1):
        assert mock_save_if_better(step, loss) is True

    assert len(os.listdir(output_dir)) == 5

    # Worse checkpoint (loss=2.0) -> not saved
    assert mock_save_if_better(6, 2.0) is False
    assert len(os.listdir(output_dir)) == 5

    # Better checkpoint (loss=0.4) -> replaces step 1 (loss=1.5)
    assert mock_save_if_better(7, 0.4) is True
    assert len(os.listdir(output_dir)) == 5
    files = os.listdir(output_dir)
    assert not any("loss_1.5000" in f for f in files)
    assert any("loss_0.4000" in f for f in files)
