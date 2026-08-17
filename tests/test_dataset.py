"""
Unit Tests for Custom Tokenizer and Dataset Packing.

This module contains unit tests validating BPE tokenizer training, French dataset loaders,
packed sliding-window sequence packing, padding conventions, and DataLoader building.
"""

import os
import tempfile

import pytest

from src.dataset import (
    PackedTextDataset,
    get_dataloader,
    is_data_prepared,
    load_french_dataset,
    load_tokenizer_for_post_training,
    prepare_and_pack_data,
    train_custom_tokenizer,
    wait_for_data_prep,
)


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


def test_dataloader_deterministic_seeding(mock_texts):
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = train_custom_tokenizer(mock_texts, vocab_size=100, save_dir=tmpdir)
        dataset = PackedTextDataset(mock_texts, tokenizer, max_seq_len=8, stride=2)
        dl1 = get_dataloader(dataset, batch_size=2, num_workers=0, pin_memory=False, shuffle=True, seed=42)
        dl2 = get_dataloader(dataset, batch_size=2, num_workers=0, pin_memory=False, shuffle=True, seed=42)

        batch1 = [b["input_ids"] for b in dl1]
        batch2 = [b["input_ids"] for b in dl2]

        assert len(batch1) == len(batch2)
        for b1, b2 in zip(batch1, batch2):
            assert (b1 == b2).all()


def test_prepare_and_pack_data_sentinel(mock_texts):
    with tempfile.TemporaryDirectory() as tmpdir:
        tok_dir = os.path.join(tmpdir, "tokenizer")
        cache_bin = os.path.join(tmpdir, "cache.bin")

        assert not is_data_prepared(cache_bin, tok_dir)

        tokenizer = train_custom_tokenizer(mock_texts, vocab_size=100, save_dir=tok_dir)
        prepare_and_pack_data(mock_texts, tokenizer, cache_bin, packing_batch_size=2)

        assert os.path.exists(cache_bin)
        assert os.path.exists(cache_bin + ".ready")
        assert is_data_prepared(cache_bin, tok_dir)

        # wait_for_data_prep should return immediately when data is prepared
        wait_for_data_prep(cache_bin, tok_dir, poll_interval=1, timeout=5)


def test_wait_for_data_prep_timeout():
    with tempfile.TemporaryDirectory() as tmpdir:
        tok_dir = os.path.join(tmpdir, "tokenizer")
        cache_bin = os.path.join(tmpdir, "cache.bin")

        with pytest.raises(TimeoutError):
            wait_for_data_prep(cache_bin, tok_dir, poll_interval=0.1, timeout=0.3, log_interval=0.1)


def test_load_tokenizer_for_post_training(tmp_path, mock_texts):
    # 1. Test loading from pretraining cache
    cache_dir = str(tmp_path / "cache")
    tok_dir = os.path.join(cache_dir, "tokenizer_checkpoint")
    train_custom_tokenizer(mock_texts, vocab_size=100, save_dir=tok_dir)

    tok1 = load_tokenizer_for_post_training(
        model_id="google/gemma-3-270m-it",
        data_cache_dir=cache_dir,
    )
    assert len(tok1) >= 256

    # 2. Test loading from pretrained checkpoint directory
    ckpt_dir = str(tmp_path / "ckpt_dir")
    train_custom_tokenizer(mock_texts, vocab_size=100, save_dir=ckpt_dir)
    tok2 = load_tokenizer_for_post_training(
        model_id="google/gemma-3-270m-it",
        pretrained_model_path=ckpt_dir,
    )
    assert len(tok2) >= 256



