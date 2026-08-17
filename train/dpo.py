"""
Direct Preference Optimization (DPO) entrypoint for French Gemma 3 alignment.

Aligns model responses with human preferences using implicit reward margins
and reference model log-probability scoring.
"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from src.config import TrainingConfig
from src.dataset import GEMMA_CHAT_TEMPLATE, load_tokenizer_for_post_training
from src.dpo_dataset import (
    DPODataset,
    get_dpo_dataloader,
    load_dpo_pairs,
    normalize_dpo_pair,
)
from src.model import FrenchGemmaModel
from src.scheduler import get_cosine_warmup_scheduler
from src.trainer import save_hf_checkpoint_dir
from train.cli import parse_args_to_config
from train.strategies.dpo import DPOStrategy

logger = logging.getLogger(__name__)


def initialize_dpo_models(
    config: TrainingConfig, vocab_size: int
) -> Tuple[FrenchGemmaModel, FrenchGemmaModel]:
    """
    Initializes the policy model and frozen reference model for DPO.
    Loads the pretrained/SFT checkpoint into the policy model if configured,
    and sets up the frozen reference model (either from ref_model_id or cloned policy).
    """
    policy_model = FrenchGemmaModel(
        model_id=config.model_id,
        vocab_size=vocab_size,
        embedding_noise_std=0.0,
    )
    if config.pretrained_model_path:
        if os.path.exists(config.pretrained_model_path):
            logger.info(f"Loading policy model weights from: {config.pretrained_model_path}")
            policy_model.load_pretrained_checkpoint(config.pretrained_model_path)
        else:
            raise FileNotFoundError(
                f"Specified pretrained_model_path does not exist: {config.pretrained_model_path}"
            )
    else:
        logger.warning(
            "No local pretrained/SFT checkpoint provided via --pretrained-model-path. "
            f"Defaulting to base Gemma 3 checkpoint from HuggingFace: '{config.model_id}'."
        )

    policy_model.ensure_tokenizer_vocab_alignment(vocab_size)

    ref_model = FrenchGemmaModel(
        model_id=config.model_id,
        vocab_size=vocab_size,
        embedding_noise_std=0.0,
    )
    if config.ref_model_id:
        logger.info(f"Loading distinct reference model weights from: {config.ref_model_id}")
        if os.path.exists(config.ref_model_id):
            ref_model.load_pretrained_checkpoint(config.ref_model_id)
        else:
            raise FileNotFoundError(
                f"Specified ref_model_id checkpoint does not exist: {config.ref_model_id}"
            )
    else:
        ref_model.load_state_dict(policy_model.state_dict())

    ref_model.ensure_tokenizer_vocab_alignment(vocab_size)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return policy_model, ref_model


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
        f"DPO Training launched. Model: {config.model_id} | Device: {config.device} | "
        f"Beta: {config.dpo_beta} | Label Smoothing: {config.dpo_label_smoothing}"
    )

    tokenizer = load_tokenizer_for_post_training(
        model_id=config.model_id,
        data_cache_dir=config.data_cache_dir,
        pretrained_model_path=config.pretrained_model_path,
    )
    pairs = load_dpo_pairs(config.dataset_path if config.dataset_path != "wikimedia/wikipedia" else None)

    if pairs:
        first_pair = normalize_dpo_pair(pairs[0])
        logger.info("=" * 80)
        logger.info("ACTUAL LOADED PREFERENCE PAIR SAMPLE:")
        logger.info("-" * 80)
        logger.info(f"  [PROMPT]:\n    {first_pair[0]}")
        logger.info(f"  [CHOSEN]:\n    {first_pair[1]}")
        logger.info(f"  [REJECTED]:\n    {first_pair[2]}")
        logger.info("=" * 80)

    dpo_dataset = DPODataset(
        pairs=pairs,
        tokenizer=tokenizer,
        max_seq_len=config.max_sequence_length,
    )

    if len(dpo_dataset) == 0:
        raise ValueError(
            "DPODataset contains 0 valid preference pairs after processing and filtering. Cannot train."
        )

    dataloader = get_dpo_dataloader(
        dataset=dpo_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        seed=config.seed,
    )
    logger.info(
        f"DataLoader ready: batch_size={config.batch_size}, "
        f"gradient_accumulation_steps={config.gradient_accumulation_steps} "
        f"(effective batch size: {config.batch_size * config.gradient_accumulation_steps})."
    )

    policy_model, ref_model = initialize_dpo_models(config=config, vocab_size=len(tokenizer))
    policy_model = policy_model.to(config.device)
    ref_model = ref_model.to(config.device)

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

    device_type = "cuda" if "cuda" in str(config.device) else ("mps" if "mps" in str(config.device) else "cpu")
    amp_dtype = (
        torch.bfloat16
        if config.amp_dtype == "bfloat16"
        else (torch.float16 if config.amp_dtype == "float16" else torch.float32)
    )

    policy_model.train()
    step = 0
    accum_loss = 0.0
    accum_batches = 0
    t0 = time.time()

    os.makedirs(config.output_dir, exist_ok=True)
    best_checkpoints: List[Dict[str, Any]] = []

    def save_if_better(step_idx: int, current_loss: float) -> None:
        if config.max_checkpoints <= 0:
            return
        checkpoint_name = f"checkpoint-step-{step_idx}-loss-{current_loss:.4f}"
        checkpoint_path = os.path.join(config.output_dir, checkpoint_name)
        should_save = False
        if len(best_checkpoints) < config.max_checkpoints:
            should_save = True
        else:
            worst = max(best_checkpoints, key=lambda x: x["loss"])
            if current_loss < worst["loss"]:
                should_save = True

        if should_save:
            save_hf_checkpoint_dir(
                checkpoint_path=checkpoint_path,
                model=policy_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                global_step=step_idx,
                metrics={
                    "loss": current_loss,
                    **strategy.latest_metrics,
                },
                config=config,
                chat_template=getattr(tokenizer, "chat_template", None) or GEMMA_CHAT_TEMPLATE,
            )
            best_checkpoints.append({"path": checkpoint_path, "loss": current_loss, "step": step_idx})
            logger.info(f"Saved new best DPO checkpoint (loss={current_loss:.4f}): {checkpoint_path}")

            if len(best_checkpoints) > config.max_checkpoints:
                worst_to_delete = max(best_checkpoints, key=lambda x: x["loss"])
                worst_path = worst_to_delete["path"]
                out_resolved = Path(config.output_dir).resolve()
                worst_resolved = Path(worst_path).resolve()
                if worst_resolved.is_relative_to(out_resolved) and os.path.exists(worst_path):
                    try:
                        if os.path.isdir(worst_path):
                            shutil.rmtree(worst_path)
                        else:
                            os.remove(worst_path)
                        logger.info(
                            f"Removed worst DPO checkpoint ({worst_to_delete['loss']:.4f}) to maintain "
                            f"top {config.max_checkpoints}: {worst_path}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint {worst_path}: {e}")
                best_checkpoints.remove(worst_to_delete)

    step_start_time = time.time()
    warmup_steps = config.warmup_steps or int(config.max_steps * (config.warmup_ratio or 0.03))
    logger.info(
        f"Starting DPO alignment loop for {config.max_steps} steps "
        f"(Warmup: {warmup_steps} steps | Initial LR: {scheduler.get_last_lr()[0]:.2e} -> "
        f"Target: {config.learning_rate:.2e} | Log interval: every {config.log_interval} steps | "
        f"Eval/Save interval: every {config.eval_interval} steps)..."
    )
    try:
        while step < config.max_steps:
            for batch in dataloader:
                if step >= config.max_steps:
                    break

                prepared_batch = strategy.prepare_batch(batch, config.device)
                with torch.amp.autocast(  # type: ignore[attr-defined]
                    device_type=device_type, dtype=amp_dtype, enabled=config.amp_enabled
                ):
                    loss = strategy.compute_loss(policy_model, prepared_batch)
                    loss_scaled = loss / config.gradient_accumulation_steps

                loss_scaled.backward()  # type: ignore[no-untyped-call]
                accum_loss += loss.item()
                accum_batches += 1

                if accum_batches % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    step_loss = accum_loss / max(1, accum_batches)
                    accum_loss = 0.0
                    accum_batches = 0

                    if step == 1 or step % config.log_interval == 0 or step == config.max_steps:
                        current_lr = scheduler.get_last_lr()[0]
                        elapsed = time.time() - t0
                        step_time = time.time() - step_start_time
                        metrics = strategy.latest_metrics
                        reward_acc = metrics.get("reward_accuracy", 0.0)
                        reward_margin = metrics.get("reward_margin", 0.0)
                        phase = f"Warmup ({step}/{warmup_steps})" if step <= warmup_steps else "Training"
                        logger.info(
                            f"Step {step}/{config.max_steps} [{phase}] | DPO Loss: {step_loss:.4f} | "
                            f"Reward Acc: {reward_acc * 100:.1f}% | Margin: {reward_margin:.4f} | "
                            f"LR: {current_lr:.2e} | Step Time: {step_time:.2f}s | Total Elapsed: {elapsed:.1f}s"
                        )
                    step_start_time = time.time()

                    # Combined eval and save interval
                    if step % config.eval_interval == 0 or step == config.max_steps:
                        save_if_better(step, step_loss)
    finally:
        logger.info(f"DPO alignment completed. Top {len(best_checkpoints)} checkpoints retained in {config.output_dir}")


if __name__ == "__main__":
    main()
