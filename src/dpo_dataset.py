"""
Direct Preference Optimization (DPO) Dataset, Collation, and Log-Probability Helpers.

Formats (prompt, chosen, rejected) pairs using Gemma 3 turn markers and extracts
sequence-level log-probabilities for preference alignment training.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Computes sequence-level sum of token log-probabilities over unmasked labels.

    Args:
        logits: Model output tensor of shape (batch_size, seq_len, vocab_size).
        labels: Target label tensor of shape (batch_size, seq_len).
        ignore_index: Label value to mask out from log-probability sum (default -100).

    Returns:
        Tensor of shape (batch_size,) containing the sum of log probabilities.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"Logits shape {logits.shape[:-1]} must match labels shape {labels.shape}"
        )

    # Shift logits and labels by 1 for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_mask = shift_labels != ignore_index
    clamped_labels = shift_labels.masked_fill(~loss_mask, 0)

    log_probs = shift_logits.log_softmax(dim=-1)
    per_token_logps = torch.gather(
        log_probs, dim=-1, index=clamped_labels.unsqueeze(-1)
    ).squeeze(-1)

    return (per_token_logps * loss_mask).sum(dim=-1)


def format_dpo_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = 2048,
    ignore_index: int = -100,
) -> Dict[str, List[int]]:
    """
    Formats a single (prompt, chosen, rejected) triplet into tokenized inputs and prompt-masked labels.
    """
    bos_token = "<bos>"
    bos_ids = tokenizer.encode(bos_token, add_special_tokens=False) if bos_token else []

    prompt_turn = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    prompt_ids = tokenizer.encode(prompt_turn, add_special_tokens=False)

    prefix_ids = bos_ids + prompt_ids
    prefix_len = len(prefix_ids)

    # Chosen turn
    chosen_str = f"{chosen}<end_of_turn>\n"
    chosen_resp_ids = tokenizer.encode(chosen_str, add_special_tokens=False)
    chosen_ids = prefix_ids + chosen_resp_ids
    chosen_labels = [ignore_index] * prefix_len + list(chosen_resp_ids)

    # Rejected turn
    rejected_str = f"{rejected}<end_of_turn>\n"
    rejected_resp_ids = tokenizer.encode(rejected_str, add_special_tokens=False)
    rejected_ids = prefix_ids + rejected_resp_ids
    rejected_labels = [ignore_index] * prefix_len + list(rejected_resp_ids)

    # Truncate
    if len(chosen_ids) > max_seq_len:
        chosen_ids = chosen_ids[:max_seq_len]
        chosen_labels = chosen_labels[:max_seq_len]

    if len(rejected_ids) > max_seq_len:
        rejected_ids = rejected_ids[:max_seq_len]
        rejected_labels = rejected_labels[:max_seq_len]

    # Pad
    pad_id = tokenizer.pad_token_id or 0
    pad_chosen_len = max(0, max_seq_len - len(chosen_ids))
    if pad_chosen_len > 0:
        chosen_ids.extend([pad_id] * pad_chosen_len)
        chosen_labels.extend([ignore_index] * pad_chosen_len)

    pad_rejected_len = max(0, max_seq_len - len(rejected_ids))
    if pad_rejected_len > 0:
        rejected_ids.extend([pad_id] * pad_rejected_len)
        rejected_labels.extend([ignore_index] * pad_rejected_len)

    return {
        "chosen_input_ids": chosen_ids,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_ids,
        "rejected_labels": rejected_labels,
    }


def normalize_dpo_pair(raw: Any) -> Tuple[str, str, str]:
    """Normalizes various preference dataset formats into (prompt, chosen, rejected) strings."""
    if isinstance(raw, dict):
        if "prompt" in raw and "chosen" in raw and "rejected" in raw:
            p = raw["prompt"]
            c = raw["chosen"]
            r = raw["rejected"]
            p_str = p if isinstance(p, str) else "\n".join(m.get("content", "") for m in p)
            c_str = c if isinstance(c, str) else "\n".join(m.get("content", "") for m in c)
            r_str = r if isinstance(r, str) else "\n".join(m.get("content", "") for m in r)
            return p_str, c_str, r_str
        if "question" in raw and "response_j" in raw and "response_k" in raw:
            return str(raw["question"]), str(raw["response_j"]), str(raw["response_k"])
    raise ValueError(f"Unsupported DPO preference record: {raw}")


def batch_format_dpo_pairs(
    pairs: List[Tuple[str, str, str]],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = 2048,
    ignore_index: int = -100,
) -> List[Dict[str, List[int]]]:
    """
    Batched encoding and formatting for DPO preference pairs.
    """
    if not pairs:
        return []

    prompt_strs = [
        f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        for prompt, _, _ in pairs
    ]
    chosen_strs = [f"{chosen}<end_of_turn>\n" for _, chosen, _ in pairs]
    rejected_strs = [f"{rejected}<end_of_turn>\n" for _, _, rejected in pairs]

    all_strs = prompt_strs + chosen_strs + rejected_strs
    all_enc = tokenizer(all_strs, add_special_tokens=False)["input_ids"]

    n = len(pairs)
    p_enc = all_enc[:n]
    c_enc = all_enc[n : 2 * n]
    r_enc = all_enc[2 * n :]

    pad_id = tokenizer.pad_token_id or 0
    results: List[Dict[str, List[int]]] = []

    for p_ids, c_ids, r_ids in zip(p_enc, c_enc, r_enc):
        chosen_input = p_ids + c_ids
        chosen_labels = [ignore_index] * len(p_ids) + c_ids
        if len(chosen_input) > max_seq_len:
            chosen_input = chosen_input[:max_seq_len]
            chosen_labels = chosen_labels[:max_seq_len]
        else:
            pad_len = max_seq_len - len(chosen_input)
            chosen_input = chosen_input + [pad_id] * pad_len
            chosen_labels = chosen_labels + [ignore_index] * pad_len

        rej_input = p_ids + r_ids
        rej_labels = [ignore_index] * len(p_ids) + r_ids
        if len(rej_input) > max_seq_len:
            rej_input = rej_input[:max_seq_len]
            rej_labels = rej_labels[:max_seq_len]
        else:
            pad_len = max_seq_len - len(rej_input)
            rej_input = rej_input + [pad_id] * pad_len
            rej_labels = rej_labels + [ignore_index] * pad_len

        results.append(
            {
                "chosen_input_ids": chosen_input,
                "chosen_labels": chosen_labels,
                "rejected_input_ids": rej_input,
                "rejected_labels": rej_labels,
            }
        )

    return results


class DPODataset(Dataset[Dict[str, torch.Tensor]]):
    """
    PyTorch Dataset for Direct Preference Optimization.
    Supports batched tokenization and formatting with granular progress verbosity.
    """

    def __init__(
        self,
        pairs: List[Any],
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int = 2048,
        batch_size: int = 2000,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.items: List[Dict[str, torch.Tensor]] = []

        total_pairs = len(pairs)
        logger.info(
            f"Starting batched tokenization and formatting for {total_pairs:,} DPO preference pairs "
            f"(chunk_size={batch_size}, max_seq_len={max_seq_len})..."
        )

        t_start = time.time()
        pad_id = tokenizer.pad_token_id or 0
        skipped_count = 0

        for chunk_idx in range(0, total_pairs, batch_size):
            chunk_t0 = time.time()
            chunk_raw = pairs[chunk_idx : chunk_idx + batch_size]

            valid_triplets: List[Tuple[str, str, str]] = []
            for raw in chunk_raw:
                try:
                    p, c, r = normalize_dpo_pair(raw)
                    valid_triplets.append((p, c, r))
                except Exception as err:
                    logger.debug(f"Skipping unparseable DPO preference pair: {err}")
                    skipped_count += 1

            if valid_triplets:
                formatted_batch = batch_format_dpo_pairs(
                    pairs=valid_triplets,
                    tokenizer=tokenizer,
                    max_seq_len=max_seq_len,
                )

                for item in formatted_batch:
                    chosen_ids_t = torch.tensor(item["chosen_input_ids"], dtype=torch.long)
                    chosen_labels_t = torch.tensor(item["chosen_labels"], dtype=torch.long)
                    chosen_mask_t = (chosen_ids_t != pad_id).long()

                    rej_ids_t = torch.tensor(item["rejected_input_ids"], dtype=torch.long)
                    rej_labels_t = torch.tensor(item["rejected_labels"], dtype=torch.long)
                    rej_mask_t = (rej_ids_t != pad_id).long()

                    self.items.append(
                        {
                            "chosen_input_ids": chosen_ids_t,
                            "chosen_labels": chosen_labels_t,
                            "chosen_attention_mask": chosen_mask_t,
                            "rejected_input_ids": rej_ids_t,
                            "rejected_labels": rej_labels_t,
                            "rejected_attention_mask": rej_mask_t,
                        }
                    )

            chunk_time = time.time() - chunk_t0
            processed_so_far = min(chunk_idx + batch_size, total_pairs)
            pct = (processed_so_far / total_pairs) * 100.0
            elapsed_total = time.time() - t_start
            throughput = processed_so_far / max(1e-4, elapsed_total)

            logger.info(
                f"Tokenization Progress: [{processed_so_far:,}/{total_pairs:,}] ({pct:5.1f}%) | "
                f"Chunk Time: {chunk_time:.2f}s | Throughput: {throughput:,.1f} pairs/s | "
                f"Valid Indexed: {len(self.items):,}"
            )

        total_time = time.time() - t_start
        if len(self.items) == 0:
            raise ValueError(
                f"DPODataset contains 0 valid preference pairs after processing and filtering {total_pairs} items."
            )

        logger.info(
            f"Tokenization & formatting completed in {total_time:.2f}s "
            f"(Avg: {total_pairs / max(1e-4, total_time):,.1f} pairs/s). "
            f"Successfully indexed {len(self.items):,}/{total_pairs:,} valid preference pairs "
            f"({skipped_count:,} skipped)."
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


class DPODataCollator:
    """Collates individual DPO items into a batched dictionary."""

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "chosen_input_ids": torch.stack([f["chosen_input_ids"] for f in features]),
            "chosen_labels": torch.stack([f["chosen_labels"] for f in features]),
            "chosen_attention_mask": torch.stack([f["chosen_attention_mask"] for f in features]),
            "rejected_input_ids": torch.stack([f["rejected_input_ids"] for f in features]),
            "rejected_labels": torch.stack([f["rejected_labels"] for f in features]),
            "rejected_attention_mask": torch.stack([f["rejected_attention_mask"] for f in features]),
        }


def get_dpo_dataloader(
    dataset: DPODataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    sampler: Optional[torch.utils.data.Sampler[int]] = None,
    seed: Optional[int] = None,
) -> DataLoader[Any]:
    """Constructs a PyTorch DataLoader for batched DPO training."""
    collator = DPODataCollator()
    actual_shuffle = shuffle if sampler is None else False
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=actual_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        generator=generator,
    )
