import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

from src.config import DatasetMixEntry

logger = logging.getLogger(__name__)

DEFAULT_FRENCH_CONVERSATIONS: List[List[Dict[str, str]]] = [
    [
        {"role": "user", "content": "Bonjour, comment t'appelles-tu ?"},
        {
            "role": "assistant",
            "content": "Bonjour, je suis FrenchGemma, un LLM entraîné en français.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Peux-tu m'expliquer ce qu'est l'apprentissage automatique ?",
        },
        {
            "role": "assistant",
            "content": (
                "L'apprentissage automatique (machine learning) est une branche de l'intelligence artificielle "
                "qui permet aux ordinateurs d'apprendre à partir de données sans programmation explicite."
            ),
        },
    ],
    [
        {
            "role": "user",
            "content": "Quelle est la capitale de la France ?",
        },
        {"role": "assistant", "content": "La capitale de la France est Paris."},
    ],
]


def load_sft_conversations(data_path: Optional[str]) -> List[Any]:
    """Loads SFT conversations from JSON/JSONL file or returns default French dialogue corpus."""
    if data_path is not None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"SFT dataset file not found: {data_path}")
        logger.info(f"Loading SFT conversation data from {data_path}...")
        conversations: List[Any] = []
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    if line.strip():
                        try:
                            conversations.append(json.loads(line.strip()))
                        except json.JSONDecodeError as err:
                            raise ValueError(f"Malformed JSONL at {data_path}:{idx}: {err}") from err
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    conversations = data
                elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                    conversations = data["data"]
                else:
                    raise ValueError(f"Expected list or dictionary with 'data' list in {data_path}")
        if not conversations:
            raise ValueError(f"No conversation samples loaded from {data_path}")
        return conversations

    logger.info("Using default French conversational fine-tuning dataset.")
    return DEFAULT_FRENCH_CONVERSATIONS


def normalize_conversation(raw_item: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Normalizes different dialogue data formats into standard messages list:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    if isinstance(raw_item, list):
        messages: List[Dict[str, str]] = []
        for turn in raw_item:
            if isinstance(turn, dict):
                if "from" in turn and "value" in turn:
                    from_val = str(turn["from"]).strip().lower()
                    if from_val in ("human", "user"):
                        role = "user"
                    elif from_val in ("system",):
                        role = "system"
                    else:
                        role = "assistant"
                    messages.append({"role": role, "content": str(turn["value"])})
                elif "role" in turn and "content" in turn:
                    raw_role = str(turn["role"]).strip().lower()
                    if raw_role in ("human", "user"):
                        role = "user"
                    elif raw_role in ("system",):
                        role = "system"
                    elif raw_role in ("assistant", "model"):
                        role = "assistant"
                    else:
                        role = raw_role
                    messages.append({"role": role, "content": str(turn["content"])})
        return messages
    if isinstance(raw_item, dict):
        if "messages" in raw_item and isinstance(raw_item["messages"], list):
            return normalize_conversation(raw_item["messages"])
        if "conversations" in raw_item and isinstance(raw_item["conversations"], list):
            return normalize_conversation(raw_item["conversations"])
        if "conversation_a" in raw_item or "conversation_b" in raw_item:
            chosen = raw_item.get("chosen_model_name")
            b_name = raw_item.get("model_b_name")
            winner = str(raw_item.get("winner", "")).strip().lower()
            is_b_winner = (chosen and b_name and chosen == b_name) or winner in (
                "model_b",
                "b",
                "model_b_preferred",
                "response_b",
            )
            if is_b_winner and "conversation_b" in raw_item:
                return normalize_conversation(raw_item["conversation_b"])
            if "conversation_a" in raw_item:
                return normalize_conversation(raw_item["conversation_a"])
            if "conversation_b" in raw_item:
                return normalize_conversation(raw_item["conversation_b"])
        if "context" in raw_item and "question" in raw_item:
            ans_text = ""
            if "answers" in raw_item:
                ans = raw_item["answers"]
                if isinstance(ans, dict) and "text" in ans:
                    txt_field = ans["text"]
                    if isinstance(txt_field, list) and len(txt_field) > 0:
                        ans_text = str(txt_field[0])
                    elif isinstance(txt_field, str):
                        ans_text = txt_field
                elif isinstance(ans, list) and len(ans) > 0:
                    ans_text = str(ans[0])
                elif isinstance(ans, str):
                    ans_text = ans
            user_content = f"Contexte:\n{raw_item['context']}\n\nQuestion:\n{raw_item['question']}"
            return [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": ans_text},
            ]
        if "context" in raw_item and "qas" in raw_item:
            qas = raw_item["qas"]
            if isinstance(qas, list) and len(qas) > 0:
                first_qa = qas[0]
                q_text = first_qa.get("question", "")
                answers = first_qa.get("answers", [])
                ans_text = ""
                if isinstance(answers, list) and len(answers) > 0:
                    first_ans = answers[0]
                    if isinstance(first_ans, dict):
                        ans_text = str(first_ans.get("text", ""))
                    else:
                        ans_text = str(first_ans)
                elif isinstance(answers, dict) and "text" in answers:
                    txt_field = answers["text"]
                    if isinstance(txt_field, list) and len(txt_field) > 0:
                        ans_text = str(txt_field[0])
                    else:
                        ans_text = str(txt_field)
                user_content = f"Contexte:\n{raw_item['context']}\n\nQuestion:\n{q_text}"
                return [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": ans_text},
                ]
        if "prompt" in raw_item and ("chosen" in raw_item or "response_a" in raw_item or "response_b" in raw_item):
            chosen_resp = raw_item.get("chosen")
            if chosen_resp is None:
                winner = str(raw_item.get("winner", "model_a")).strip().lower()
                if winner in ("model_b", "b", "response_b"):
                    chosen_resp = raw_item.get("response_b") or ""
                else:
                    chosen_resp = raw_item.get("response_a") or ""
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

    loaded_per_dataset: List[List[List[Dict[str, str]]]] = []

    for entry in data_mix:
        items: List[Any] = []
        if fallback_conversations and entry.dataset_path in fallback_conversations:
            items = fallback_conversations[entry.dataset_path]
        else:
            split_to_use = entry.split if entry.split is not None else default_split
            ds_path = entry.dataset_path
            # Normalize known FQuAD aliases if needed
            if ds_path in ("almanach/fquad", "fquad"):
                ds_path = "CATIE-AQ/frenchQA"
            try:
                import datasets

                logger.info(
                    f"Loading HF dataset for SFT: {ds_path} "
                    f"(name={entry.dataset_name}, split={split_to_use})..."
                )
                if entry.dataset_name:
                    ds = datasets.load_dataset(
                        ds_path,
                        entry.dataset_name,
                        split=split_to_use,
                        verification_mode="no_checks",
                    )
                else:
                    ds = datasets.load_dataset(
                        ds_path,
                        split=split_to_use,
                        verification_mode="no_checks",
                    )
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
                if norm and any(m.get("role") in ("assistant", "model") and m.get("content", "").strip() for m in norm):
                    normalized_items.append(norm)
            except Exception:
                continue

        logger.info(f"  -> Successfully extracted {len(normalized_items)} dialogues from {entry.dataset_path}")
        loaded_per_dataset.append(normalized_items)

    rng = random.Random(seed)
    sampled_per_dataset: List[List[List[Dict[str, str]]]] = []

    # Calculate active weights over datasets that actually have samples
    active_total_pct = sum(entry.percentage for entry, items in zip(data_mix, loaded_per_dataset) if len(items) > 0)

    if is_all or target_total is None or active_total_pct <= 0:
        for ds_items in loaded_per_dataset:
            sampled_items = list(ds_items)
            rng.shuffle(sampled_items)
            sampled_per_dataset.append(sampled_items)
    else:
        for entry, ds_items in zip(data_mix, loaded_per_dataset):
            if not ds_items:
                sampled_per_dataset.append([])
                continue
            renorm_weight = entry.percentage / active_total_pct
            desired_count = int(round(target_total * renorm_weight))
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
    pad_to_max: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Encodes conversation turns into input_ids and labels with prompt masking.
    
    Returns:
        Tuple of (input_ids, labels) truncated to max_seq_len (and optionally padded).
    """
    all_input_ids: List[int] = []
    all_labels: List[int] = []

    bos_token = tokenizer.bos_token or "<bos>"
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

    if len(all_input_ids) > max_seq_len:
        all_input_ids = all_input_ids[:max_seq_len]
        all_labels = all_labels[:max_seq_len]

    if pad_to_max:
        pad_id = tokenizer.pad_token_id or 0
        pad_len = max(0, max_seq_len - len(all_input_ids))
        if pad_len > 0:
            all_input_ids.extend([pad_id] * pad_len)
            all_labels.extend([ignore_index] * pad_len)

    return all_input_ids, all_labels


def batch_format_messages_with_prompt_mask(
    conversations: List[List[Dict[str, str]]],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int = 2048,
    ignore_index: int = -100,
) -> List[Tuple[List[int], List[int]]]:
    """
    Batched encoding of conversation turns into input_ids and labels with prompt masking.
    Batches all turn strings across conversations into a single tokenizer call to maximize throughput.

    Returns:
        List of (input_ids, labels) tuples, one per input conversation.
    """
    if not conversations:
        return []

    bos_token = tokenizer.bos_token or "<bos>"
    bos_ids = tokenizer.encode(bos_token, add_special_tokens=False) if bos_token else []
    model_prefix_str = "<start_of_turn>model\n"
    model_prefix_ids = tokenizer.encode(model_prefix_str, add_special_tokens=False)

    all_turn_strings: List[str] = []
    # Plan tracks per conversation: list of (is_masked: bool, is_static: bool, static_ids: List[int], turn_idx: int)
    conv_plans: List[List[Tuple[bool, bool, List[int], int]]] = []

    for messages in conversations:
        plan: List[Tuple[bool, bool, List[int], int]] = []
        if bos_ids:
            plan.append((True, True, bos_ids, -1))

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                turn_str = f"<start_of_turn>system\n{content}<end_of_turn>\n"
                turn_idx = len(all_turn_strings)
                all_turn_strings.append(turn_str)
                plan.append((True, False, [], turn_idx))
            elif role == "user":
                turn_str = f"<start_of_turn>user\n{content}<end_of_turn>\n"
                turn_idx = len(all_turn_strings)
                all_turn_strings.append(turn_str)
                plan.append((True, False, [], turn_idx))
            elif role in ("model", "assistant"):
                plan.append((True, True, model_prefix_ids, -1))
                turn_str = f"{content}<end_of_turn>\n"
                turn_idx = len(all_turn_strings)
                all_turn_strings.append(turn_str)
                plan.append((False, False, [], turn_idx))

        conv_plans.append(plan)

    if all_turn_strings:
        batch_encodings = tokenizer(all_turn_strings, add_special_tokens=False)["input_ids"]
    else:
        batch_encodings = []

    results: List[Tuple[List[int], List[int]]] = []
    for plan in conv_plans:
        input_ids: List[int] = []
        labels: List[int] = []
        for is_masked, is_static, static_ids, turn_idx in plan:
            t_ids = static_ids if is_static else batch_encodings[turn_idx]
            input_ids.extend(t_ids)
            if is_masked:
                labels.extend([ignore_index] * len(t_ids))
            else:
                labels.extend(t_ids)

        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            labels = labels[:max_seq_len]

        results.append((input_ids, labels))

    return results


class SFTDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    PyTorch Dataset for Supervised Fine-Tuning with turn-level prompt masking.
    Supports batched tokenization and formatting with granular progress verbosity.
    """

    def __init__(
        self,
        conversations: List[Any],
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int = 2048,
        batch_size: int = 2000,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.items: List[Dict[str, torch.Tensor]] = []

        total_conversations = len(conversations)
        logger.info(
            f"Starting batched tokenization and formatting for {total_conversations:,} SFT conversation samples "
            f"(chunk_size={batch_size}, max_seq_len={max_seq_len})..."
        )

        t_start = time.time()
        total_active_tokens = 0
        total_masked_tokens = 0
        skipped_count = 0

        for chunk_idx in range(0, total_conversations, batch_size):
            chunk_t0 = time.time()
            chunk_raw = conversations[chunk_idx : chunk_idx + batch_size]

            valid_messages_list: List[List[Dict[str, str]]] = []
            for raw in chunk_raw:
                try:
                    msgs = normalize_conversation(raw)
                    if msgs:
                        valid_messages_list.append(msgs)
                    else:
                        skipped_count += 1
                except Exception as err:
                    logger.debug(f"Skipping unparseable SFT conversation item: {err}")
                    skipped_count += 1

            if valid_messages_list:
                encoded_pairs = batch_format_messages_with_prompt_mask(
                    conversations=valid_messages_list,
                    tokenizer=tokenizer,
                    max_seq_len=max_seq_len,
                )

                for input_ids, labels in encoded_pairs:
                    active_tokens = sum(1 for lbl in labels[1:] if lbl != -100)
                    if active_tokens == 0:
                        skipped_count += 1
                        continue

                    masked_tokens = len(labels) - active_tokens
                    total_active_tokens += active_tokens
                    total_masked_tokens += masked_tokens

                    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
                    labels_t = torch.tensor(labels, dtype=torch.long)
                    attention_mask_t = torch.ones_like(input_ids_t)

                    self.items.append(
                        {
                            "input_ids": input_ids_t,
                            "labels": labels_t,
                            "attention_mask": attention_mask_t,
                        }
                    )

            chunk_time = time.time() - chunk_t0
            processed_so_far = min(chunk_idx + batch_size, total_conversations)
            pct = (processed_so_far / total_conversations) * 100.0
            elapsed_total = time.time() - t_start
            throughput = processed_so_far / max(1e-4, elapsed_total)

            logger.info(
                f"Tokenization Progress: [{processed_so_far:,}/{total_conversations:,}] ({pct:5.1f}%) | "
                f"Chunk Time: {chunk_time:.2f}s | Throughput: {throughput:,.1f} convs/s | "
                f"Valid Indexed: {len(self.items):,} | Active Tokens: {total_active_tokens:,}"
            )

        total_time = time.time() - t_start
        if len(self.items) == 0:
            raise ValueError(
                f"SFTDataset contains 0 valid conversational samples after processing and filtering "
                f"{total_conversations} input items."
            )

        logger.info(
            f"Tokenization & formatting completed in {total_time:.2f}s "
            f"(Avg: {total_conversations / max(1e-4, total_time):,.1f} convs/s). "
            f"Successfully indexed {len(self.items):,}/{total_conversations:,} valid conversational samples "
            f"({total_active_tokens:,} active response tokens, {total_masked_tokens:,} prompt-masked tokens, "
            f"{skipped_count:,} skipped)."
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


class SFTDataCollator:
    """
    Collates a list of SFT dataset items into a batch tensor dictionary,
    supporting dynamic batch-level padding up to the maximum length in the batch.
    """

    def __init__(self, pad_token_id: int = 0, ignore_index: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        max_len = max(f["input_ids"].size(0) for f in features)

        batch_input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        batch_labels = torch.full((batch_size, max_len), self.ignore_index, dtype=torch.long)
        batch_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, f in enumerate(features):
            seq_len = f["input_ids"].size(0)
            batch_input_ids[i, :seq_len] = f["input_ids"]
            batch_labels[i, :seq_len] = f["labels"]
            batch_attention_mask[i, :seq_len] = f["attention_mask"]

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
    pad_id = dataset.tokenizer.pad_token_id if dataset.tokenizer and dataset.tokenizer.pad_token_id is not None else 0
    collator = SFTDataCollator(pad_token_id=pad_id)
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
