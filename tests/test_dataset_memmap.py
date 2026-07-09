"""
Unit Tests for numpy.memmap Binary Dataset Caching and Iterator Tokenizer Training.
"""

import os
import tempfile

import numpy as np
import pytest
from transformers import PreTrainedTokenizerFast

from src.dataset import PackedTextDataset, train_custom_tokenizer


def test_train_tokenizer_from_iterator():
    # Define generator
    def text_generator():
        yield "Bonjour, comment ça va aujourd'hui?"
        yield "Le grand chat noir dort sous la table basse."
        yield "Entraîner un modèle de langue sur le français est fascinant."

    with tempfile.TemporaryDirectory() as tmpdir:
        # Train custom tokenizer using generator
        tokenizer = train_custom_tokenizer(
            texts=text_generator(),
            vocab_size=1000,
            save_dir=tmpdir,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        )

        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        assert tokenizer.bos_token == "<bos>"
        assert tokenizer.eos_token == "<eos>"
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.padding_side == "right"


def test_packed_text_dataset_memmap():
    texts = [
        "Un petit exemple de texte en français.",
        "Un autre exemple pour remplir la base de données.",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Train Tokenizer
        tokenizer = train_custom_tokenizer(
            texts=texts,
            vocab_size=1000,
            save_dir=tmpdir,
        )

        # 2. Tokenize and write to a binary file on disk (uint32)
        bin_path = os.path.join(tmpdir, "dataset.bin")
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            doc_ids = []
            if bos_id is not None:
                doc_ids.append(bos_id)
            doc_ids.extend(tokens)
            if eos_id is not None:
                doc_ids.append(eos_id)
            all_tokens.extend(doc_ids)

        arr = np.array(all_tokens, dtype=np.uint32)
        with open(bin_path, "wb") as f:
            f.write(arr.tobytes())

        # 3. Load with PackedTextDataset using bin_path
        max_seq_len = 8
        stride = 2
        dataset = PackedTextDataset(
            bin_path=bin_path,
            max_seq_len=max_seq_len,
            stride=stride,
            tokenizer=tokenizer,
        )

        # Verify dataset length
        step = max_seq_len - stride
        expected_chunks = (len(all_tokens) - max_seq_len) // step + 1
        assert len(dataset) == expected_chunks

        # Verify getitem returns correct shape and values
        first_item = dataset[0]
        assert "input_ids" in first_item
        assert len(first_item["input_ids"]) == max_seq_len
        assert first_item["input_ids"] == all_tokens[:max_seq_len]

        second_item = dataset[1]
        assert second_item["input_ids"] == all_tokens[step : step + max_seq_len]


def test_packed_text_dataset_memmap_padding():
    # Test case where total tokens < max_seq_len
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = train_custom_tokenizer(
            texts=["Bonjour."],
            vocab_size=500,
            save_dir=tmpdir,
        )

        bin_path = os.path.join(tmpdir, "small_dataset.bin")
        bos_id = tokenizer.bos_token_id or 1
        eos_id = tokenizer.eos_token_id or 2
        tokens = tokenizer.encode("Bonjour.", add_special_tokens=False)
        all_tokens = [bos_id] + tokens + [eos_id]

        arr = np.array(all_tokens, dtype=np.uint32)
        with open(bin_path, "wb") as f:
            f.write(arr.tobytes())

        max_seq_len = 32
        dataset = PackedTextDataset(
            bin_path=bin_path,
            max_seq_len=max_seq_len,
            stride=5,
            tokenizer=tokenizer,
        )

        assert len(dataset) == 1
        item = dataset[0]
        assert len(item["input_ids"]) == max_seq_len
        # The beginning should match the actual tokens
        assert item["input_ids"][:len(all_tokens)] == all_tokens
        # The rest should be padding
        pad_id = tokenizer.pad_token_id or 0
        assert item["input_ids"][len(all_tokens):] == [pad_id] * (max_seq_len - len(all_tokens))


def test_packed_text_dataset_memmap_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "empty_dataset.bin")
        # Create empty file
        open(bin_path, "wb").close()

        with pytest.raises(ValueError, match="is empty. Cannot initialize PackedTextDataset"):
            PackedTextDataset(
                bin_path=bin_path,
                max_seq_len=8,
                stride=2,
            )
