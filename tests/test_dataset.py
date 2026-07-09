"""
Unit Tests for Custom Tokenizer and Dataset Packing.

This module contains unit tests validating BPE tokenizer training, French dataset loaders,
packed sliding-window sequence packing, padding conventions, and DataLoader building.
"""

import tempfile

import pytest

from src.dataset import PackedTextDataset, get_dataloader, load_french_dataset, train_custom_tokenizer


@pytest.fixture
def mock_texts():
    return [
        "Un petit texte en français.",
        "Un autre document pour tester la tokenisation.",
        "Le modèle de langue sera entraîné sur ces exemples.",
    ]


def test_tokenizer_training(mock_texts):
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = train_custom_tokenizer(mock_texts, vocab_size=100, save_dir=tmpdir)
        assert len(tokenizer) >= 256  # Byte-level includes at least 256 bytes
        assert tokenizer.bos_token == "<bos>"
        assert tokenizer.eos_token == "<eos>"
        assert tokenizer.pad_token == "<pad>"

        # Test encode / decode
        encoded = tokenizer.encode("français")
        decoded = tokenizer.decode(encoded)
        assert "français" in decoded


def test_packed_text_dataset(mock_texts):
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = train_custom_tokenizer(mock_texts, vocab_size=100, save_dir=tmpdir)

        # Sequence length 8, overlap 2
        dataset = PackedTextDataset(mock_texts, tokenizer, max_seq_len=8, stride=2)
        assert len(dataset) > 0

        item = dataset[0]
        assert "input_ids" in item
        assert len(item["input_ids"]) == 8
        assert item["input_ids"][0] == tokenizer.bos_token_id


def test_dataloader_batching(mock_texts):
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = train_custom_tokenizer(mock_texts, vocab_size=100, save_dir=tmpdir)
        dataset = PackedTextDataset(mock_texts, tokenizer, max_seq_len=10, stride=3)
        dl = get_dataloader(dataset, batch_size=2, num_workers=0, pin_memory=False)

        batch = next(iter(dl))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

        # Batch size 2, seq len 10
        assert batch["input_ids"].shape == (2, 10)
        assert batch["labels"].shape == (2, 10)


def test_load_french_dataset_fallback():
    # Calling load_french_dataset with invalid path to trigger fallback
    texts = load_french_dataset(dataset_path="invalid_path_xyz", fallback_texts=["fallback"])
    assert texts == ["fallback"]
