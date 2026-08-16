"""
Direct Preference Optimization (DPO) entrypoint for French Gemma 3 alignment.

Aligns model responses with human preferences using implicit reward margins
and reference model log-probability scoring.
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
from src.dpo_dataset import DPODataset, get_dpo_dataloader
from src.model import FrenchGemmaModel
from src.scheduler import get_cosine_warmup_scheduler
from train.cli import parse_args_to_config
from train.strategies.dpo import DPOStrategy

logger = logging.getLogger(__name__)

DEFAULT_FRENCH_DPO_PAIRS = [
    {
        "prompt": "Bonjour, comment t'appelles-tu ?",
        "chosen": "Bonjour, je suis FrenchGemma, un LLM entraîné en français. Comment puis-je t'aider ?",
        "rejected": "Je suis un robot sans nom.",
    },
    {
        "prompt": "Explique-moi la théorie de la relativité simplement.",
        "chosen": (
            "La relativité d'Einstein montre que l'espace et le temps sont liés et que "
            "le temps s'écoule différemment selon la vitesse et la gravité."
        ),
        "rejected": "C'est une théorie très compliquée de physique que personne ne comprend.",
    },
    {
        "prompt": "Donne un conseil pour apprendre une nouvelle langue.",
        "chosen": (
            "Pratique quotidiennement, même 15 minutes, et immerge-toi avec des podcasts, "
            "des livres et des conversations réelles."
        ),
        "rejected": "Lis juste un dictionnaire entier par cœur.",
    },
]


def load_dpo_pairs(data_path: Optional[str] = None) -> List[Any]:
    """Loads DPO preference pairs from JSON/JSONL file or returns default French preference pairs."""
    if data_path is not None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"DPO dataset file not found: {data_path}")
        logger.info(f"Loading DPO preference data from {data_path}...")
        pairs: List[Any] = []
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    if line.strip():
                        try:
                            pairs.append(json.loads(line.strip()))
                        except json.JSONDecodeError as err:
                            raise ValueError(f"Malformed JSONL at {data_path}:{idx}: {err}") from err
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    pairs = data
                elif isinstance(data, dict) and "data" in data:
                    pairs = data["data"]
        if not pairs:
            raise ValueError(f"No preference pairs loaded from {data_path}")
        return pairs

    logger.info("Using default French preference alignment dataset.")
    return DEFAULT_FRENCH_DPO_PAIRS


def main() -> None:
    """Main CLI entrypoint for French Gemma 3 Direct Preference Optimization."""
    config: TrainingConfig = parse_args_to_config(modality="dpo")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
        level=logging.INFO,
    )

    logger.info(
        f"DPO Alignment launched. Model: {config.model_id} | Beta: {config.dpo_beta} | "
        f"Device: {config.device} | Max Seq Len: {config.max_sequence_length}"
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

    dpo_data_path = (
        config.dataset_path
        if config.dataset_path and config.dataset_path != "wikimedia/wikipedia"
        else None
    )
    pairs = load_dpo_pairs(dpo_data_path)
    dpo_dataset = DPODataset(
        pairs=pairs,
        tokenizer=tokenizer,
        max_seq_len=config.max_sequence_length,
    )

    dataloader = get_dpo_dataloader(
        dataset=dpo_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        seed=config.seed,
    )

    logger.info("Initializing Policy Model for DPO...")
    policy_model = FrenchGemmaModel(
        model_id=config.model_id,
        vocab_size=len(tokenizer),
        embedding_noise_std=0.0,
    ).to(config.device)

    logger.info("Initializing Frozen Reference Model for DPO...")
    ref_model = FrenchGemmaModel(
        model_id=config.model_id,
        vocab_size=len(tokenizer),
        embedding_noise_std=0.0,
    ).to(config.device)
    ref_model.load_state_dict(policy_model.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    strategy = DPOStrategy(
        ref_model=ref_model,
        beta=config.dpo_beta,
        label_smoothing=config.dpo_label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_warmup_scheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
        eta_min_ratio=config.eta_min_ratio,
    )

    policy_model.train()
    step = 0
    t0 = time.time()

    logger.info(f"Starting DPO alignment loop for {config.max_steps} steps...")
    while step < config.max_steps:
        for batch in dataloader:
            if step >= config.max_steps:
                break

            prepared_batch = strategy.prepare_batch(batch, config.device)
            optimizer.zero_grad(set_to_none=True)

            loss = strategy.compute_loss(policy_model, prepared_batch)
            loss.backward()  # type: ignore[no-untyped-call]

            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            step += 1

            if step % config.log_interval == 0 or step == config.max_steps:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                metrics = strategy.latest_metrics
                reward_acc = metrics.get("reward_accuracy", 0.0)
                reward_margin = metrics.get("reward_margin", 0.0)
                logger.info(
                    f"Step {step}/{config.max_steps} | DPO Loss: {loss.item():.4f} | "
                    f"Reward Acc: {reward_acc * 100:.1f}% | Margin: {reward_margin:.4f} | "
                    f"LR: {current_lr:.2e} | Elapsed: {elapsed:.1f}s"
                )

    os.makedirs(config.output_dir, exist_ok=True)
    save_path = os.path.join(config.output_dir, f"dpo_checkpoint_step_{step}.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": policy_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        save_path,
    )
    logger.info(f"DPO alignment completed. Model checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()
