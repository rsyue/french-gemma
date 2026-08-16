"""
Supervised Fine-Tuning (SFT) entrypoint for French Gemma 3 models.

Executes turn-based conversational fine-tuning with prompt masking
using Gemma 3 turn tokens (<bos>, <eos>, <start_of_turn>, <end_of_turn>).
"""

import json
import logging
import os
import time
from typing import Any, List, Optional

import torch
from transformers import AutoTokenizer

from src.config import TrainingConfig
from src.dataset import train_custom_tokenizer
from src.model import FrenchGemmaModel
from src.scheduler import get_cosine_warmup_scheduler
from src.sft_dataset import SFTDataset, get_sft_dataloader
from train.builder import TrainingFactory
from train.cli import parse_args_to_config

logger = logging.getLogger(__name__)

DEFAULT_FRENCH_CONVERSATIONS = [
    [
        {"role": "user", "content": "Bonjour, comment t'appelles-tu ?"},
        {
            "role": "assistant",
            "content": "Bonjour, je suis FrenchGemma, un LLM entraîné en français. Comment puis-je t'aider ?",
        },
    ],
    [
        {"role": "user", "content": "Qu'est-ce que l'intelligence artificielle ?"},
        {
            "role": "assistant",
            "content": (
                "L'intelligence artificielle désigne des systèmes capables d'accomplir des tâches qui "
                "demandent habituellement l'intelligence humaine, comme comprendre le langage et apprendre."
            ),
        },
    ],
    [
        {"role": "user", "content": "Donne-moi une recette simple de crêpes."},
        {
            "role": "assistant",
            "content": (
                "Pour des crêpes faciles : 250g de farine, 4 œufs, 500ml de lait, "
                "un peu de sucre et une pincée de sel. "
                "Mélange le tout et fais cuire dans une poêle bien chaude !"
            ),
        },
    ],
]


def load_sft_conversations(data_path: Optional[str] = None) -> List[Any]:
    """Loads SFT conversations from JSON/JSONL file or returns default French dialogue corpus."""
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading SFT conversation data from {data_path}...")
        conversations: List[Any] = []
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        conversations.append(json.loads(line.strip()))
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    conversations = data
                elif isinstance(data, dict) and "data" in data:
                    conversations = data["data"]
        return conversations

    logger.info("Using default French conversational fine-tuning dataset.")
    return DEFAULT_FRENCH_CONVERSATIONS


def main() -> None:
    """Main CLI entrypoint for French Gemma 3 Supervised Fine-Tuning."""
    config: TrainingConfig = parse_args_to_config()

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

    conversations = load_sft_conversations()
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
    model = FrenchGemmaModel(
        model_id=config.model_id,
        vocab_size=len(tokenizer),
        embedding_noise_std=config.embedding_noise_std,
    ).to(config.device)

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

    model.train()
    step = 0
    t0 = time.time()

    logger.info(f"Starting SFT optimization loop for {config.max_steps} steps...")
    while step < config.max_steps:
        for batch in dataloader:
            if step >= config.max_steps:
                break

            prepared_batch = strategy.prepare_batch(batch, config.device)
            optimizer.zero_grad(set_to_none=True)

            loss = strategy.compute_loss(model, prepared_batch)
            loss.backward()  # type: ignore[no-untyped-call]

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            step += 1

            if step % config.log_interval == 0 or step == config.max_steps:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                logger.info(
                    f"Step {step}/{config.max_steps} | Loss: {loss.item():.4f} | "
                    f"LR: {current_lr:.2e} | Elapsed: {elapsed:.1f}s"
                )

    os.makedirs(config.output_dir, exist_ok=True)
    save_path = os.path.join(config.output_dir, f"sft_checkpoint_step_{step}.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        save_path,
    )
    logger.info(f"SFT training completed. Model checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()
