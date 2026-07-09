"""
Tokenizer Training and Dataset Packing.

This module provides utilities to train a custom ByteLevelBPETokenizer on unsupervised texts,
load French text datasets, pack tokens with a sliding window stride, and build PyTorch DataLoaders.
"""

import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class PackedTextDataset(Dataset[Dict[str, Any]]):
    """
    A PyTorch Dataset that tokenizes and packs text documents into sequences of
    max_sequence_length, separated by BOS and EOS tokens, using a sliding window overlap (stride).
    """

    def __init__(
        self, texts: List[str], tokenizer: PreTrainedTokenizerFast, max_seq_len: int, stride: int = 50
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

        t0 = time.time()
        total_texts = len(texts)
        logger.info(f"Starting tokenization of {total_texts} documents...")
        print(f"Starting tokenization of {total_texts} documents...")

        # Tokenize and format each document as [BOS] + tokens + [EOS]
        all_token_ids = []
        processed_count = 0
        for idx, text in enumerate(texts):
            if not text.strip():
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            doc_ids = []
            if bos_id is not None:
                doc_ids.append(bos_id)
            doc_ids.extend(tokens)
            if eos_id is not None:
                doc_ids.append(eos_id)
            all_token_ids.extend(doc_ids)
            processed_count += 1

            # Print/log progress periodically if dataset is large (e.g. >= 100 texts)
            if total_texts >= 100 and ((idx + 1) % max(1, total_texts // 10) == 0 or (idx + 1) == total_texts):
                elapsed = time.time() - t0
                throughput = (idx + 1) / elapsed if elapsed > 0 else 0
                progress_pct = ((idx + 1) / total_texts) * 100
                msg = (
                    f"Tokenization progress: {idx + 1}/{total_texts} documents "
                    f"({progress_pct:.1f}%) | Speed: {throughput:.2f} docs/sec"
                )
                logger.info(msg)
                print(msg)

        t_tokenize = time.time() - t0
        msg = (
            f"Tokenization completed in {t_tokenize:.2f} seconds. "
            f"Processed {processed_count} non-empty documents. "
            f"Total tokens: {len(all_token_ids)}"
        )
        logger.info(msg)
        print(msg)

        # Pack token ids into chunks using sliding window
        logger.info("Packing tokens into fixed-length sequences...")
        print("Packing tokens into fixed-length sequences...")
        t_pack_start = time.time()
        self.chunks = []
        step = max(1, max_seq_len - stride)

        # If total tokens are less than max_seq_len, pad to max_seq_len
        if len(all_token_ids) < max_seq_len:
            pad_id = tokenizer.pad_token_id or 0
            padded_ids = all_token_ids + [pad_id] * (max_seq_len - len(all_token_ids))
            self.chunks.append(padded_ids)
        else:
            total_tokens = len(all_token_ids)
            num_steps = (total_tokens - max_seq_len) // step + 1
            log_interval = max(1, num_steps // 10)
            for chunk_idx, i in enumerate(range(0, total_tokens - max_seq_len + 1, step)):
                self.chunks.append(all_token_ids[i : i + max_seq_len])
                if num_steps >= 1000 and ((chunk_idx + 1) % log_interval == 0 or (chunk_idx + 1) == num_steps):
                    pack_pct = ((chunk_idx + 1) / num_steps) * 100
                    msg = f"Packing progress: {chunk_idx + 1}/{num_steps} sequences packed ({pack_pct:.1f}%)"
                    logger.info(msg)
                    print(msg)

        t_pack = time.time() - t_pack_start
        msg = (
            f"Packing completed in {t_pack:.2f} seconds. "
            f"Generated {len(self.chunks)} sequences of length {max_seq_len} "
            f"(stride={stride}, step={step})."
        )
        logger.info(msg)
        print(msg)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        chunk = self.chunks[idx]
        return {
            "input_ids": chunk,
        }


def train_custom_tokenizer(
    texts: List[str],
    vocab_size: int = 32000,
    save_dir: str = "./tokenizer_checkpoint",
    special_tokens: Optional[List[str]] = None,
) -> PreTrainedTokenizerFast:
    """
    Trains a custom ByteLevelBPETokenizer on provided texts and saves it as a HuggingFace PreTrainedTokenizerFast.
    """
    logger.info(f"Training custom tokenizer on {len(texts)} texts, vocab_size={vocab_size}...")
    print(f"Training custom tokenizer on {len(texts)} texts, vocab_size={vocab_size}...")
    t0 = time.time()
    os.makedirs(save_dir, exist_ok=True)
    if special_tokens is None:
        special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    # Write texts to a temporary file for the tokenizer trainer
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
        temp_file_path = f.name

    try:
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=[temp_file_path], vocab_size=vocab_size, min_frequency=2, special_tokens=special_tokens)

        # Save tokenizer
        tokenizer_json_path = os.path.join(save_dir, "tokenizer.json")
        tokenizer.save(tokenizer_json_path)

        # Wrap as PreTrainedTokenizerFast
        hf_tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
            tokenizer_file=tokenizer_json_path,
            bos_token="<bos>" if "<bos>" in special_tokens else None,
            eos_token="<eos>" if "<eos>" in special_tokens else None,
            pad_token="<pad>" if "<pad>" in special_tokens else None,
            unk_token="<unk>" if "<unk>" in special_tokens else None,
        )

        # Ensure correct padding side for training decoders (Right padding)
        hf_tokenizer.padding_side = "right"

        # Ensure pad_token_id, etc. are correctly assigned
        hf_tokenizer.pad_token_id = special_tokens.index("<pad>") if "<pad>" in special_tokens else 0
        hf_tokenizer.bos_token_id = special_tokens.index("<bos>") if "<bos>" in special_tokens else 1
        hf_tokenizer.eos_token_id = special_tokens.index("<eos>") if "<eos>" in special_tokens else 2
        hf_tokenizer.unk_token_id = special_tokens.index("<unk>") if "<unk>" in special_tokens else 3

        # Save the wrapped tokenizer configuration
        hf_tokenizer.save_pretrained(save_dir)
        t_train = time.time() - t0
        logger.info(f"Tokenizer training completed in {t_train:.2f} seconds. Saved to {save_dir}")
        print(f"Tokenizer training completed in {t_train:.2f} seconds. Saved to {save_dir}")
        return hf_tokenizer
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def load_french_dataset(
    dataset_path: str = "wikimedia/wikipedia",
    dataset_name: str = "20231101.fr",
    split: str = "train[:1000]",
    fallback_texts: Optional[List[str]] = None,
) -> List[str]:
    """
    Loads French unsupervised dataset using HuggingFace datasets library.
    Falls back to a mock corpus or provided texts if loading fails or for unit testing.
    """
    try:
        # Load the dataset
        ds = load_dataset(dataset_path, dataset_name, split=split)
        # Extract text column
        if "text" in ds.column_names:
            return ds["text"]  # type: ignore[no-any-return]
        elif "content" in ds.column_names:
            return ds["content"]  # type: ignore[no-any-return]
        else:
            # Fallback to the first string column
            str_cols = [col for col, val in ds.features.items() if hasattr(val, "dtype") and val.dtype == "string"]
            if str_cols:
                return ds[str_cols[0]]  # type: ignore[no-any-return]
            return [str(row) for row in ds]
    except Exception as e:
        print(f"Warning: Failed to load dataset {dataset_path}/{dataset_name}: {e}")
        if fallback_texts:
            return fallback_texts
        # Default mock French corpus
        return [
            "Le français est une langue romane parlée principalement en France.",
            "L'apprentissage profond est une technique d'intelligence artificielle.",
            "Le modèle Gemma est développé par Google.",
            "Cette bibliothèque permet de pré-entraîner un modèle sur du texte brut.",
            "Les jetons sont découpés au niveau de l'octet pour gérer les élisions du français.",
            "Les architectures modernes de réseaux de neurones utilisent le mécanisme d'attention.",
        ]


def get_dataloader(
    dataset: PackedTextDataset,
    batch_size: int,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader[Any]:
    """
    Creates a PyTorch DataLoader utilizing DataCollatorForLanguageModeling.
    """
    msg = (
        f"Creating DataLoader: batch_size={batch_size}, shuffle={shuffle}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}"
    )
    logger.info(msg)
    print(msg)
    collator = DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm=False)

    # Prefetch factor must be None if num_workers is 0
    actual_prefetch = prefetch_factor if num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=actual_prefetch,
        pin_memory=pin_memory,
        collate_fn=collator,
    )
