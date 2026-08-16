"""
Unit and integration tests for Gemma 3 / ChatML turn tokens, SFT prompt masking,
SFT dataset collation, and SFT training strategy.
"""

import pytest
import torch
import torch.nn as nn

from src.dataset import train_custom_tokenizer
from src.sft_dataset import (
    SFTDataset,
    format_messages_with_prompt_mask,
    get_sft_dataloader,
)
from train.builder import TrainingFactory
from train.strategies.sft import SFTStrategy


@pytest.fixture
def mock_tokenizer(tmp_path):
    sample_texts = [
        "<start_of_turn>user\nBonjour, comment vas-tu?<end_of_turn>\n"
        "<start_of_turn>model\nJe vais très bien, merci!<end_of_turn>\n",
        "<start_of_turn>user\nExplique l'apprentissage profond.<end_of_turn>\n"
        "<start_of_turn>model\nL'apprentissage profond est une sous-branche de l'IA.<end_of_turn>\n",
    ]
    tok_dir = str(tmp_path / "tokenizer")
    tokenizer = train_custom_tokenizer(sample_texts, vocab_size=200, save_dir=tok_dir)
    return tokenizer


def test_tokenizer_registers_gemma_turn_tokens(mock_tokenizer):
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
    for token in special_tokens:
        token_id = mock_tokenizer.convert_tokens_to_ids(token)
        assert token_id is not None
        if token != "<unk>":
            assert token_id != mock_tokenizer.unk_token_id, f"Token {token} mapped to unk_token_id"

    assert mock_tokenizer.chat_template is not None
    assert "<start_of_turn>" in mock_tokenizer.chat_template
    assert "<end_of_turn>" in mock_tokenizer.chat_template


def test_chat_template_rendering(mock_tokenizer):
    messages = [
        {"role": "user", "content": "Bonjour, comment t'appelles-tu ?"},
        {"role": "assistant", "content": "Bonjour, je suis FrenchGemma, un LLM entraîné en français."},
    ]
    rendered = mock_tokenizer.apply_chat_template(messages, tokenize=False)
    expected = (
        "<bos><start_of_turn>user\nBonjour, comment t'appelles-tu ?<end_of_turn>\n"
        "<start_of_turn>model\nBonjour, je suis FrenchGemma, un LLM entraîné en français.<end_of_turn>\n"
    )
    assert rendered == expected


def test_chat_template_with_system_message(mock_tokenizer):
    messages = [
        {"role": "system", "content": "Tu es un assistant utile."},
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Bonjour ! Comment puis-je t'aider ?"},
    ]
    rendered = mock_tokenizer.apply_chat_template(messages, tokenize=False)
    assert "<start_of_turn>system\nTu es un assistant utile.<end_of_turn>\n" in rendered
    assert "<start_of_turn>user\nBonjour<end_of_turn>\n" in rendered
    assert "<start_of_turn>model\nBonjour ! Comment puis-je t'aider ?<end_of_turn>\n" in rendered


def test_format_messages_with_prompt_mask(mock_tokenizer):
    messages = [
        {"role": "user", "content": "Bonjour"},
        {"role": "model", "content": "Bonjour, comment allez-vous?"},
    ]
    input_ids, labels = format_messages_with_prompt_mask(
        messages=messages,
        tokenizer=mock_tokenizer,
        max_seq_len=64,
    )

    assert len(input_ids) == 64
    assert len(labels) == 64

    # Prompt tokens should have label -100
    prompt_str = "<bos><start_of_turn>user\nBonjour<end_of_turn>\n<start_of_turn>model\n"
    prompt_len = len(mock_tokenizer.encode(prompt_str, add_special_tokens=False))
    for i in range(prompt_len):
        assert labels[i] == -100

    # Model response tokens should match input_ids
    response_tokens = mock_tokenizer.encode("Bonjour, comment allez-vous?<end_of_turn>\n", add_special_tokens=False)
    for j, token_id in enumerate(response_tokens):
        pos = prompt_len + j
        assert input_ids[pos] == token_id
        assert labels[pos] == token_id

    # Padding positions should have label -100
    pad_id = mock_tokenizer.pad_token_id or 0
    total_tokens = prompt_len + len(response_tokens)
    for k in range(total_tokens, 64):
        assert input_ids[k] == pad_id
        assert labels[k] == -100


def test_sft_dataset_and_collator(mock_tokenizer):
    conversations = [
        [
            {"role": "user", "content": "Question 1"},
            {"role": "model", "content": "Réponse 1"},
        ],
        {"instruction": "Instruction 2", "response": "Réponse 2"},
    ]

    dataset = SFTDataset(conversations=conversations, tokenizer=mock_tokenizer, max_seq_len=32)
    assert len(dataset) == 2

    item0 = dataset[0]
    assert "input_ids" in item0
    assert "labels" in item0
    assert "attention_mask" in item0
    assert item0["input_ids"].shape == (32,)
    assert item0["labels"].shape == (32,)
    assert item0["attention_mask"].shape == (32,)

    loader = get_sft_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 32)
    assert batch["labels"].shape == (2, 32)
    assert batch["attention_mask"].shape == (2, 32)


def test_sft_strategy_compute_loss():
    strategy = SFTStrategy()

    class MockModel(nn.Module):
        def forward(self, input_ids, labels=None, attention_mask=None):
            class Out:
                loss = torch.tensor(1.234, requires_grad=True)
            return Out()

    model = MockModel()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "labels": torch.tensor([[-100, 2, 3], [-100, -100, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }

    loss = strategy.compute_loss(model, batch)
    assert isinstance(loss, torch.Tensor)
    assert abs(loss.item() - 1.234) < 1e-4


def test_training_factory_builds_sft_strategy():
    strategy = TrainingFactory.build_strategy("sft")
    assert isinstance(strategy, SFTStrategy)


def test_format_multi_turn_messages_with_prompt_mask(mock_tokenizer):
    messages = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Réponse 1"},
        {"role": "user", "content": "Question 2"},
        {"role": "assistant", "content": "Réponse 2"},
    ]

    input_ids, labels = format_messages_with_prompt_mask(
        messages=messages,
        tokenizer=mock_tokenizer,
        max_seq_len=128,
    )

    # Decode and check that user turns have label -100 and assistant turns have active labels
    assert len(input_ids) == 128
    assert len(labels) == 128

    # First turn prompt should be masked
    u1_prefix = "<bos><start_of_turn>user\nQuestion 1<end_of_turn>\n<start_of_turn>model\n"
    u1_len = len(mock_tokenizer.encode(u1_prefix, add_special_tokens=False))
    for i in range(u1_len):
        assert labels[i] == -100

    # First assistant response should be unmasked
    a1_str = "Réponse 1<end_of_turn>\n"
    a1_len = len(mock_tokenizer.encode(a1_str, add_special_tokens=False))
    for j in range(a1_len):
        pos = u1_len + j
        assert labels[pos] == input_ids[pos]


def test_normalize_conversation_sharegpt():
    from src.sft_dataset import normalize_conversation

    sharegpt_item = [
        {"from": "human", "value": "Bonjour !"},
        {"from": "gpt", "value": "Salut !"},
    ]
    normalized = normalize_conversation(sharegpt_item)
    assert len(normalized) == 2
    assert normalized[0] == {"role": "user", "content": "Bonjour !"}
    assert normalized[1] == {"role": "assistant", "content": "Salut !"}


def test_sft_strategy_compute_loss_raw_logits_fallback():
    strategy = SFTStrategy()

    class RawLogitsModel(nn.Module):
        def forward(self, input_ids, labels=None, attention_mask=None):
            # Return raw tensor of logits (batch=2, seq=3, vocab=5)
            logits = torch.zeros(2, 3, 5)
            # Make target token have higher logit
            for i in range(2):
                for j in range(2):
                    logits[i, j, 2] = 5.0
            return logits

    model = RawLogitsModel()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "labels": torch.tensor([[-100, 2, 3], [-100, -100, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }

    loss = strategy.compute_loss(model, batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0


def test_load_sft_conversations_file_handling(tmp_path):
    import json

    from train.sft import load_sft_conversations

    # Non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_sft_conversations(str(tmp_path / "missing.jsonl"))

    # Valid JSON file
    json_path = tmp_path / "sft.json"
    data = [
        [
            {"role": "user", "content": "Test user"},
            {"role": "assistant", "content": "Test assistant"},
        ]
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    loaded = load_sft_conversations(str(json_path))
    assert len(loaded) == 1
    assert loaded[0][0]["content"] == "Test user"


def test_sft_checkpoint_rotation_logic(tmp_path):
    import os

    output_dir = str(tmp_path / "checkpoints_sft")
    os.makedirs(output_dir, exist_ok=True)
    best_checkpoints = []
    max_checkpoints = 5

    def mock_save_if_better(step_idx: int, current_loss: float):
        checkpoint_name = f"sft_checkpoint_step_{step_idx}_loss_{current_loss:.4f}.pt"
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
                f.write("mock_checkpoint")
            best_checkpoints.append({"path": checkpoint_path, "loss": current_loss, "step": step_idx})
        return should_save

    # Save 5 checkpoints
    for step, loss in enumerate([2.5, 2.2, 2.0, 1.8, 1.6], start=1):
        assert mock_save_if_better(step, loss) is True

    assert len(os.listdir(output_dir)) == 5

    # Worse checkpoint (loss=3.0) -> not saved
    assert mock_save_if_better(6, 3.0) is False
    assert len(os.listdir(output_dir)) == 5

    # Better checkpoint (loss=1.2) -> replaces step 1 (loss=2.5)
    assert mock_save_if_better(7, 1.2) is True
    assert len(os.listdir(output_dir)) == 5
    files = os.listdir(output_dir)
    assert not any("loss_2.5000" in f for f in files)
    assert any("loss_1.2000" in f for f in files)
