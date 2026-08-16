"""
Gemma 3 / ChatML Supervised Fine-Tuning (SFT) Dataset and Collation.

This module formats multi-turn chat dialogues using Gemma 3 turn tokens (<start_of_turn>, <end_of_turn>)
and applies prompt loss masking (-100 for user/system tokens) so that cross-entropy loss is computed
strictly on assistant/model response tokens.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

from src.config import DatasetMixEntry

logger = logging.getLogger(__name__)


def normalize_conversation(raw_item: Union[List[Dict[str, str]], Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Normalizes different dialogue data formats into standard messages list:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    if isinstance(raw_item, list):
        messages: List[Dict[str, str]] = []
        for turn in raw_item:
            if isinstance(turn, dict):
                if "from" in turn and "value" in turn:
                    role = "user" if turn["from"] in ("human", "user") else "assistant"
                    messages.append({"role": role, "content": str(turn["value"])})
                elif "role" in turn and "content" in turn:
                    raw_role = str(turn["role"])
                    role = "user" if raw_role in ("human", "user") else (
                        "assistant" if raw_role in ("assistant", "model") else raw_role
                    )
                    messages.append({"role": role, "content": str(turn["content"])})
        return messages
    if isinstance(raw_item, dict):
        if "messages" in raw_item and isinstance(raw_item["messages"], list):
            return normalize_conversation(raw_item["messages"])
        if "conversations" in raw_item and isinstance(raw_item["conversations"], list):
            return normalize_conversation(raw_item["conversations"])
        if "context" in raw_item and "question" in raw_item:
            ans_text = ""
            if "answers" in raw_item:
                ans = raw_item["answers"]
                if isinstance(ans, dict) and "text" in ans and len(ans["text"]) > 0:
                    ans_text = str(ans["text"][0])
                elif isinstance(ans, list) and len(ans) > 0:
                    ans_text = str(ans[0])
                elif isinstance(ans, str):
                    ans_text = ans
            user_content = f"Contexte:\n{raw_item['context']}\n\nQuestion:\n{raw_item['question']}"
            return [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": ans_text},
            ]
        if "prompt" in raw_item and ("chosen" in raw_item or "response_a" in raw_item or "response_b" in raw_item):
            chosen_resp = raw_item.get("chosen")
            if chosen_resp is None:
                winner = raw_item.get("winner", "model_a")
                if winner == "model_b":
                    chosen_resp = raw_item.get("response_b", "")
                else:
                    chosen_resp = raw_item.get("response_a", "")
            return [
                {"role": "user", "content": str(raw_item["prompt"])},
                {"role": "assistant", "content": str(chosen_resp)},
            ]
        if "instruction" in raw_item and "response" in raw_item:
            return [
                {"role": "user", "content": str(raw_item["instruction"])},
                {"role": "assistant", "content": str(raw_item["response"])},
            ]
        if "prompt" in raw_item and "response" in raw_item:
            return [
                {"role": "user", "content": str(raw_item["prompt"])},
                {"role": "assistant", "content": str(raw_item["response"])},
            ]
    raise ValueError(f"Unsupported conversation record format: {type(raw_item).__name__}")


def load_sft_dataset_mix(
    data_mix: List[DatasetMixEntry],
    total_examples: Union[int, str] = "all",
    default_split: str = "train",
    seed: int = 42,
    fallback_conversations: Optional[Dict[str, List[Any]]] = None,
) -> List[List[Dict[str, str]]]:
    """
    Loads and mixes conversational datasets according to configured percentages.
    """
    if not data_mix:
        raise ValueError("data_mix cannot be empty.")

    total_pct = sum(entry.percentage for entry in data_mix)
    for entry in data_mix:
        if entry.percentage < 0:
            raise ValueError(
                f"Dataset percentage must be non-negative, got {entry.percentage} for {entry.dataset_path}"
            )
    if total_pct <= 0:
        raise ValueError("Total percentage sum must be greater than zero.")

    is_all = False
    target_total: Optional[int] = None
    if isinstance(total_examples, str):
        val_str = total_examples.strip().lower()
        if val_str in ("all", "full", "none", "0"):
            is_all = True
        elif val_str.isdigit():
            target_total = int(val_str)
        else:
            is_all = True
    elif isinstance(total_examples, int):
        if total_examples <= 0:
            is_all = True
        else:
            target_total = total_examples

    normalized_weights = [entry.percentage / total_pct for entry in data_mix]
    loaded_per_dataset: List[List[List[Dict[str, str]]]] = []

    for entry in data_mix:
        items: List[Any] = []
        if fallback_conversations and entry.dataset_path in fallback_conversations:
            items = fallback_conversations[entry.dataset_path]
        else:
            split_to_use = entry.split if entry.split is not None else default_split
            try:
                import datasets

                logger.info(
                    f"Loading HF dataset for SFT: {entry.dataset_path} "
                    f"(name={entry.dataset_name}, split={split_to_use})..."
                )
                if entry.dataset_name:
                    ds = datasets.load_dataset(entry.dataset_path, entry.dataset_name, split=split_to_use)
                else:
                    ds = datasets.load_dataset(entry.dataset_path, split=split_to_use)
                items = list(ds)
            except Exception as err:
                logger.warning(
                    f"Failed to load dataset {entry.dataset_path}: {err}. Checking fallbacks..."
                )
                if fallback_conversations and entry.dataset_path in fallback_conversations:
                    items = fallback_conversations[entry.dataset_path]
                else:
                    items = []

        normalized_items: List[List[Dict[str, str]]] = []
        for raw in items:
            try:
                norm = normalize_conversation(raw)
                if norm:
                    normalized_items.append(norm)
            except Exception:
                continue

        loaded_per_dataset.append(normalized_items)

    rng = random.Random(seed)
    sampled_per_dataset: List[List[List[Dict[str, str]]]] = []

    if is_all or target_total is None:
        for ds_items in loaded_per_dataset:
            sampled_items = list(ds_items)
            rng.shuffle(sampled_items)
            sampled_per_dataset.append(sampled_items)
    else:
        for ds_items, weight in zip(loaded_per_dataset, normalized_weights):
            desired_count = int(round(target_total * weight))
            if not ds_items:
                sampled_per_dataset.append([])
                continue
            if len(ds_items) >= desired_count:
                indices = rng.sample(range(len(ds_items)), desired_count)
                sampled_per_dataset.append([ds_items[i] for i in indices])
            else:
                sampled_items = list(ds_items)
                rng.shuffle(sampled_items)
                sampled_per_dataset.append(sampled_items)

    all_mixed: List[List[Dict[str, str]]] = []
    for sampled in sampled_per_dataset:
        all_mixed.extend(sampled)

    rng.shuffle(all_mixed)
    if target_total is not None and not is_all and len(all_mixed) > target_total:
        all_mixed = all_mixed[:target_total]

    logger.info(f"Successfully loaded and mixed {len(all_mixed)} SFT conversation samples.")
    return all_mixed


def format_messages_with_prompt_mask(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = 2048,
    ignore_index: int = -100,
) -> Tuple[List[int], List[int]]:
    """
    Encodes conversation turns into input_ids and labels with prompt masking.
    
    Returns:
        Tuple of (input_ids, labels) padded/truncated to max_seq_len.
    """
    all_input_ids: List[int] = []
    all_labels: List[int] = []

    bos_token = "<bos>"
    bos_ids = tokenizer.encode(bos_token, add_special_tokens=False) if bos_token else []
    all_input_ids.extend(bos_ids)
    all_labels.extend([ignore_index] * len(bos_ids))

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "system":
            turn_str = f"<start_of_turn>system\n{content}<end_of_turn>\n"
            turn_ids = tokenizer.encode(turn_str, add_special_tokens=False)
            all_input_ids.extend(turn_ids)
            all_labels.extend([ignore_index] * len(turn_ids))
        elif role == "user":
            turn_str = f"<start_of_turn>user\n{content}<end_of_turn>\n"
            turn_ids = tokenizer.encode(turn_str, add_special_tokens=False)
            all_input_ids.extend(turn_ids)
            all_labels.extend([ignore_index] * len(turn_ids))
        elif role in ("model", "assistant"):
            prefix_str = "<start_of_turn>model\n"
            prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
            all_input_ids.extend(prefix_ids)
            all_labels.extend([ignore_index] * len(prefix_ids))

            content_str = f"{content}<end_of_turn>\n"
            content_ids = tokenizer.encode(content_str, add_special_tokens=False)
            all_input_ids.extend(content_ids)
            all_labels.extend(content_ids)

    # Truncate to max_seq_len
    if len(all_input_ids) > max_seq_len:
        all_input_ids = all_input_ids[:max_seq_len]
        all_labels = all_labels[:max_seq_len]

    # Pad to max_seq_len
    pad_id = tokenizer.pad_token_id or 0
    pad_len = max(0, max_seq_len - len(all_input_ids))
    if pad_len > 0:
        all_input_ids.extend([pad_id] * pad_len)
        all_labels.extend([ignore_index] * pad_len)

    return all_input_ids, all_labels


class SFTDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    PyTorch Dataset for Supervised Fine-Tuning with turn-level prompt masking.
    """

    def __init__(
        self,
        conversations: List[Any],
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.items: List[Dict[str, torch.Tensor]] = []

        logger.info(f"Formatting {len(conversations)} SFT conversation samples with prompt masking...")
        for raw in conversations:
            messages = normalize_conversation(raw)
            input_ids, labels = format_messages_with_prompt_mask(
                messages=messages,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
            )
            input_ids_t = torch.tensor(input_ids, dtype=torch.long)
            labels_t = torch.tensor(labels, dtype=torch.long)
            pad_id = tokenizer.pad_token_id or 0
            attention_mask_t = (input_ids_t != pad_id).long()

            self.items.append(
                {
                    "input_ids": input_ids_t,
                    "labels": labels_t,
                    "attention_mask": attention_mask_t,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


class SFTDataCollator:
    """Collates a list of SFT dataset items into a batch tensor dictionary."""

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = torch.stack([f["input_ids"] for f in features])
        batch_labels = torch.stack([f["labels"] for f in features])
        batch_attention_mask = torch.stack([f["attention_mask"] for f in features])
        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "attention_mask": batch_attention_mask,
        }


def get_sft_dataloader(
    dataset: SFTDataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    sampler: Optional[torch.utils.data.Sampler[int]] = None,
    seed: Optional[int] = None,
) -> DataLoader[Any]:
    """Builds a PyTorch DataLoader for SFT dataset batches."""
    collator = SFTDataCollator()
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
