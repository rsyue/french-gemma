"""
Supervised Fine-Tuning (SFT) entrypoint for French Gemma 3 models.

Executes turn-based conversational fine-tuning with prompt masking
using Gemma 3 turn tokens (<bos>, <eos>, <start_of_turn>, <end_of_turn>).
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from src.config import TrainingConfig
from src.dataset import train_custom_tokenizer
from src.model import FrenchGemmaModel
from src.scheduler import get_cosine_warmup_scheduler
from src.sft_dataset import SFTDataset, get_sft_dataloader, load_sft_dataset_mix
from train.builder import TrainingFactory
from train.cli import parse_args_to_config

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


def initialize_sft_model(config: TrainingConfig, vocab_size: int) -> FrenchGemmaModel:
    """
    Initializes the French Gemma model for SFT.
    If config.pretrained_model_path is set and exists, loads the local pretrained checkpoint.
    Otherwise, logs a warning and defaults to the base Gemma 3 checkpoint from HuggingFace.
    """
    model = FrenchGemmaModel(
        model_id=config.model_id,
        vocab_size=vocab_size,
        embedding_noise_std=config.embedding_noise_std,
    )
    if config.pretrained_model_path:
        if os.path.exists(config.pretrained_model_path):
            logger.info(f"Loading local pretrained checkpoint from: {config.pretrained_model_path}")
            model.load_pretrained_checkpoint(config.pretrained_model_path)
        else:
            raise FileNotFoundError(
                f"Specified pretrained_model_path does not exist: {config.pretrained_model_path}"
            )
    else:
        logger.warning(
            "No local pretrained checkpoint provided via --pretrained-model-path. "
            f"Defaulting to base Gemma 3 checkpoint from HuggingFace: '{config.model_id}'."
        )
    return model


def main() -> None:
    """Main CLI entrypoint for French Gemma 3 Supervised Fine-Tuning."""
    config: TrainingConfig = parse_args_to_config(modality="sft")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
        level=logging.INFO,
    )

    logger.info(
        f"SFT Training launched. Model: {config.model_id} | Device: {config.device} | "
        f"Max Seq Len: {config.max_sequence_length}"
    )

    tokenizer_dir = os.path.join(config.data_cache_dir, "tokenizer_checkpoint")
    if os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
        logger.info(f"Loading existing tokenizer from {tokenizer_dir}")
        tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        logger.info("Training initial tokenizer with Gemma 3 turn tokens...")
        seed_texts = [
            "<start_of_turn>user\nBonjour, comment t'appelles-tu ?<end_of_turn>\n"
            "<start_of_turn>model\nBonjour, je suis FrenchGemma, un LLM entraîné en français.<end_of_turn>\n",
            "Texte d'entraînement français pour initialisation du tokenizer.",
        ]
        tokenizer = train_custom_tokenizer(
            seed_texts,
            vocab_size=config.vocab_size,
            save_dir=tokenizer_dir,
        )

    if config.data_mix:
        logger.info(f"Loading SFT conversation dataset mix ({len(config.data_mix)} entries)...")
        conversations = load_sft_dataset_mix(
            data_mix=config.data_mix,
            total_examples=config.num_examples,
            seed=config.seed,
        )
    else:
        sft_data_path = (
            config.dataset_path
            if config.dataset_path and config.dataset_path != "wikimedia/wikipedia"
            else None
        )
        conversations = load_sft_conversations(sft_data_path)

    if not conversations:
        logger.warning("No conversations loaded from data source. Falling back to default French dialogues.")
        conversations = DEFAULT_FRENCH_CONVERSATIONS

    sft_dataset = SFTDataset(
        conversations=conversations,
        tokenizer=tokenizer,
        max_seq_len=config.max_sequence_length,
    )

    dataloader = get_sft_dataloader(
        dataset=sft_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        seed=config.seed,
    )

    logger.info("Initializing French Gemma model for SFT...")
    model = initialize_sft_model(config=config, vocab_size=len(tokenizer)).to(config.device)

    strategy = TrainingFactory.build_strategy("sft")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_warmup_scheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
        eta_min_ratio=config.eta_min_ratio,
    )

    device_type = "cuda" if "cuda" in str(config.device) else ("mps" if "mps" in str(config.device) else "cpu")
    amp_dtype = (
        torch.bfloat16
        if config.amp_dtype == "bfloat16"
        else (torch.float16 if config.amp_dtype == "float16" else torch.float32)
    )

    model.train()
    step = 0
    accum_loss = 0.0
    accum_batches = 0
    t0 = time.time()

    os.makedirs(config.output_dir, exist_ok=True)
    best_checkpoints: List[Dict[str, Any]] = []

    def save_if_better(step_idx: int, current_loss: float) -> None:
        if config.max_checkpoints <= 0:
            return
        checkpoint_name = f"sft_checkpoint_step_{step_idx}_loss_{current_loss:.4f}.pt"
        checkpoint_path = os.path.join(config.output_dir, checkpoint_name)
        should_save = False
        if len(best_checkpoints) < config.max_checkpoints:
            should_save = True
        else:
            worst = max(best_checkpoints, key=lambda x: x["loss"])
            if current_loss < worst["loss"]:
                should_save = True

        if should_save:
            torch.save(
                {
                    "step": step_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": current_loss,
                    "config": config,
                },
                checkpoint_path,
            )
            best_checkpoints.append({"path": checkpoint_path, "loss": current_loss, "step": step_idx})
            logger.info(f"Saved new best SFT checkpoint (loss={current_loss:.4f}): {checkpoint_path}")

            if len(best_checkpoints) > config.max_checkpoints:
                worst_to_delete = max(best_checkpoints, key=lambda x: x["loss"])
                worst_path = worst_to_delete["path"]
                out_resolved = Path(config.output_dir).resolve()
                worst_resolved = Path(worst_path).resolve()
                if worst_resolved.is_relative_to(out_resolved) and os.path.exists(worst_path):
                    try:
                        os.remove(worst_path)
                        logger.info(
                            f"Removed worst SFT checkpoint ({worst_to_delete['loss']:.4f}) to maintain "
                            f"top {config.max_checkpoints}: {worst_path}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint {worst_path}: {e}")
                best_checkpoints.remove(worst_to_delete)

    logger.info(f"Starting SFT optimization loop for {config.max_steps} steps...")
    try:
        while step < config.max_steps:
            for batch in dataloader:
                if step >= config.max_steps:
                    break

                prepared_batch = strategy.prepare_batch(batch, config.device)
                with torch.amp.autocast(  # type: ignore[attr-defined]
                    device_type=device_type, dtype=amp_dtype, enabled=config.amp_enabled
                ):
                    loss = strategy.compute_loss(model, prepared_batch)
                    loss_scaled = loss / config.gradient_accumulation_steps

                loss_scaled.backward()  # type: ignore[no-untyped-call]
                accum_loss += loss.item()
                accum_batches += 1

                if accum_batches % config.gradient_accumulation_steps == 0 or (step + 1) >= config.max_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    step_loss = accum_loss / max(1, accum_batches)
                    accum_loss = 0.0
                    accum_batches = 0

                    if step % config.log_interval == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        elapsed = time.time() - t0
                        logger.info(
                            f"Step {step}/{config.max_steps} | Loss: {step_loss:.4f} | "
                            f"LR: {current_lr:.2e} | Elapsed: {elapsed:.1f}s"
                        )

                    # Combined eval and save interval
                    if step % config.eval_interval == 0 or step == config.max_steps:
                        save_if_better(step, step_loss)
    finally:
        logger.info(f"SFT training completed. Top {len(best_checkpoints)} checkpoints retained in {config.output_dir}")


if __name__ == "__main__":
    main()
