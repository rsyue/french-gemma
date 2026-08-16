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


def test_batch_format_messages_with_prompt_mask_parity(mock_tokenizer):
    from src.sft_dataset import batch_format_messages_with_prompt_mask

    convs = [
        [
            {"role": "system", "content": "Système d'assistance."},
            {"role": "user", "content": "Bonjour"},
            {"role": "model", "content": "Bonjour !"},
        ],
        [
            {"role": "user", "content": "Quelle est la capitale de la France ?"},
            {"role": "model", "content": "Paris."},
            {"role": "user", "content": "Merci !"},
            {"role": "model", "content": "De rien !"},
        ],
    ]

    batched = batch_format_messages_with_prompt_mask(convs, mock_tokenizer, max_seq_len=64)
    assert len(batched) == 2

    for i, msgs in enumerate(convs):
        single_ids, single_labels = format_messages_with_prompt_mask(
            messages=msgs,
            tokenizer=mock_tokenizer,
            max_seq_len=64,
            pad_to_max=False,
        )
        assert batched[i][0] == single_ids
        assert batched[i][1] == single_labels


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


def test_normalize_conversation_all_formats():
    from src.sft_dataset import normalize_conversation

    # 1. Luciole format (messages list)
    luciole_item = {
        "messages": [
            {"role": "user", "content": "Quelle est la vitesse de la lumière ?"},
            {"role": "assistant", "content": "Environ 300 000 km/s dans le vide."},
        ]
    }
    norm1 = normalize_conversation(luciole_item)
    assert len(norm1) == 2
    assert norm1[0]["role"] == "user"
    assert norm1[1]["content"] == "Environ 300 000 km/s dans le vide."

    # 2. Comparia format (prompt + winner / responses)
    comparia_item_a = {
        "prompt": "Explique la photosynthèse.",
        "response_a": "C'est la production d'énergie par les plantes via la lumière.",
        "response_b": "Les plantes absorbent l'eau.",
        "winner": "model_a",
    }
    norm2 = normalize_conversation(comparia_item_a)
    assert len(norm2) == 2
    assert norm2[0]["role"] == "user"
    assert norm2[1]["content"] == "C'est la production d'énergie par les plantes via la lumière."

    comparia_item_b = {
        "prompt": "Qui a peint la Joconde ?",
        "response_a": "Michel-Ange",
        "response_b": "Léonard de Vinci",
        "winner": "model_b",
    }
    norm2_b = normalize_conversation(comparia_item_b)
    assert norm2_b[1]["content"] == "Léonard de Vinci"

    # 3. FQuAD format (context + question + answers)
    fquad_item = {
        "context": "Paris est la capitale et la plus grande ville de France.",
        "question": "Quelle est la capitale de la France ?",
        "answers": {"text": ["Paris"], "answer_start": [0]},
    }
    norm3 = normalize_conversation(fquad_item)
    assert len(norm3) == 2
    assert "Contexte:\nParis est la capitale" in norm3[0]["content"]
    assert "Question:\nQuelle est la capitale" in norm3[0]["content"]
    assert norm3[1]["content"] == "Paris"


def test_load_pretrained_checkpoint_in_model(tmp_path):
    import os

    from src.model import FrenchGemmaModel

    model_id = "google/gemma-3-270m-it"
    vocab_size = 256

    model_src = FrenchGemmaModel(model_id=model_id, vocab_size=vocab_size)
    ckpt_dir = str(tmp_path / "pretrained_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    weights_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    torch.save(model_src.state_dict(), weights_path)

    model_dst = FrenchGemmaModel(model_id=model_id, vocab_size=vocab_size)
    model_dst.load_pretrained_checkpoint(ckpt_dir)

    # Verify weights are identical
    for p1, p2 in zip(model_src.parameters(), model_dst.parameters()):
        assert torch.allclose(p1, p2)


def test_sft_pretrained_checkpoint_warning_and_loading(tmp_path, caplog):
    import logging
    import os

    from src.config import TrainingConfig
    from src.model import FrenchGemmaModel
    from train.sft import initialize_sft_model

    # Case 1: No pretrained model path passed -> emits warning
    config_default = TrainingConfig(model_id="google/gemma-3-270m-it", pretrained_model_path=None)
    with caplog.at_level(logging.WARNING):
        model1 = initialize_sft_model(config_default, vocab_size=256)
    assert any("Defaulting to base Gemma 3 checkpoint from HuggingFace" in record.message for record in caplog.records)
    assert isinstance(model1, FrenchGemmaModel)

    # Case 2: Local pretrained model path passed -> loads checkpoint
    ckpt_dir = str(tmp_path / "ckpt_pretrain")
    os.makedirs(ckpt_dir, exist_ok=True)
    src_model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=256)
    torch.save(src_model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))

    config_local = TrainingConfig(model_id="google/gemma-3-270m-it", pretrained_model_path=ckpt_dir)
    model2 = initialize_sft_model(config_local, vocab_size=256)
    assert isinstance(model2, FrenchGemmaModel)
    for p1, p2 in zip(src_model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_load_sft_dataset_mix_proportions():
    from src.config import DatasetMixEntry
    from src.sft_dataset import load_sft_dataset_mix

    data_mix = [
        DatasetMixEntry(
            dataset_path="OpenLLM-France/Luciole-PostTraining-Dataset-1.1",
            dataset_name="sft_instruct",
            percentage=40.0,
        ),
        DatasetMixEntry(
            dataset_path="ministere-culture/comparia-votes",
            percentage=30.0,
        ),
        DatasetMixEntry(
            dataset_path="almanach/fquad",
            percentage=30.0,
        ),
    ]

    fallback_samples = {
        "OpenLLM-France/Luciole-PostTraining-Dataset-1.1": [
            [
                {"role": "user", "content": f"Luciole prompt {i}"},
                {"role": "assistant", "content": f"Luciole resp {i}"},
            ]
            for i in range(100)
        ],
        "ministere-culture/comparia-votes": [
            [
                {"role": "user", "content": f"Comparia prompt {i}"},
                {"role": "assistant", "content": f"Comparia resp {i}"},
            ]
            for i in range(100)
        ],
        "almanach/fquad": [
            [
                {"role": "user", "content": f"FQuAD prompt {i}"},
                {"role": "assistant", "content": f"FQuAD resp {i}"},
            ]
            for i in range(100)
        ],
    }

    mixed = load_sft_dataset_mix(data_mix, total_examples=100, fallback_conversations=fallback_samples, seed=42)
    assert len(mixed) == 100
    luciole_count = sum(1 for m in mixed if "Luciole" in m[0]["content"])
    comparia_count = sum(1 for m in mixed if "Comparia" in m[0]["content"])
    fquad_count = sum(1 for m in mixed if "FQuAD" in m[0]["content"])

    assert luciole_count == 40
    assert comparia_count == 30
    assert fquad_count == 30


def test_sft_dataset_skips_malformed_and_empty_assistant(mock_tokenizer):
    # Conversations containing 1 valid dialogue, 1 unparseable format, and 1 user-only dialogue
    raw_data = [
        {"role": "invalid_structure"},
        [
            {"role": "user", "content": "Question without response"},
        ],
        [
            {"role": "user", "content": "Bonjour !"},
            {"role": "assistant", "content": "Bonjour !"},
        ],
    ]
    ds = SFTDataset(conversations=raw_data, tokenizer=mock_tokenizer, max_seq_len=128)
    # Only the valid dialogue with assistant response should be retained
    assert len(ds) == 1
    item = ds[0]
    assert any(label_val != -100 for label_val in item["labels"].tolist())


def test_sft_dynamic_collator():
    from src.sft_dataset import SFTDataCollator

    collator = SFTDataCollator(pad_token_id=0, ignore_index=-100)
    features = [
        {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "labels": torch.tensor([-100, 2, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([4, 5], dtype=torch.long),
            "labels": torch.tensor([-100, 5], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        },
    ]
    batch = collator(features)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)
    # Second sample is padded by 1 token
    assert batch["input_ids"][1, 2].item() == 0
    assert batch["labels"][1, 2].item() == -100
    assert batch["attention_mask"][1, 2].item() == 0


def test_load_sft_dataset_mix_renormalizes_on_missing_dataset():
    from src.config import DatasetMixEntry
    from src.sft_dataset import load_sft_dataset_mix

    data_mix = [
        DatasetMixEntry(dataset_path="dataset_a", percentage=40.0),
        DatasetMixEntry(dataset_path="dataset_b_gated", percentage=60.0),
    ]

    # Only dataset_a has samples (simulating dataset_b failing to load)
    fallback_samples = {
        "dataset_a": [
            [{"role": "user", "content": f"A {i}"}, {"role": "assistant", "content": f"A resp {i}"}]
            for i in range(50)
        ],
        "dataset_b_gated": [],
    }

    mixed = load_sft_dataset_mix(data_mix, total_examples=20, fallback_conversations=fallback_samples, seed=42)
    # Should draw all 20 samples from dataset_a
    assert len(mixed) == 20
    assert all("A" in m[0]["content"] for m in mixed)


def test_load_pretrained_checkpoint_errors(tmp_path):
    import os

    from src.model import FrenchGemmaModel

    model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=256)

    # Error on non-existent path
    with pytest.raises(FileNotFoundError):
        model.load_pretrained_checkpoint(str(tmp_path / "non_existent"))

    # Error on empty directory
    empty_dir = str(tmp_path / "empty_dir")
    os.makedirs(empty_dir, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        model.load_pretrained_checkpoint(empty_dir)

