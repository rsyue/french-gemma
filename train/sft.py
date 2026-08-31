"""
Supervised Fine-Tuning (SFT) entrypoint for French Gemma 3 models.

Executes turn-based conversational fine-tuning with prompt masking
using Gemma 3 turn tokens (<bos>, <eos>, <start_of_turn>, <end_of_turn>).
"""

import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.config import TrainingConfig
from src.dataset import load_tokenizer_for_post_training
from src.model import FrenchGemmaModel
from src.scheduler import get_cosine_warmup_scheduler
from src.sft_dataset import (
    DEFAULT_FRENCH_CONVERSATIONS,
    SFTDataset,
    format_messages_with_prompt_mask,
    get_sft_dataloader,
    load_sft_conversations,
    load_sft_dataset_mix,
)
from train.builder import TrainingFactory
from train.cli import parse_args_to_config

logger = logging.getLogger(__name__)


def initialize_sft_model(config: TrainingConfig, vocab_size: int) -> FrenchGemmaModel:
    """
    Initializes the French Gemma model for SFT.
    If config.pretrained_model_path is set and exists, loads the local pretrained checkpoint.
    Otherwise, logs a warning and defaults to the base Gemma 3 checkpoint from HuggingFace.
    Ensures that the tokenizer length corresponds to the size of the final linear layer.
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

    # Ensure length of tokenizer corresponds to final linear layer and embedding dimensions
    model.ensure_tokenizer_vocab_alignment(vocab_size)
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

    tokenizer: Any = load_tokenizer_for_post_training(
        model_id=config.model_id,
        data_cache_dir=config.data_cache_dir,
        pretrained_model_path=config.pretrained_model_path,
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

    # Display an actual sample conversation from the loaded dataset for verification
    if conversations:
        sample_conv = conversations[0]
        logger.info("=" * 80)
        logger.info("ACTUAL LOADED DATASET SAMPLE (Conversational Turns):")
        logger.info("-" * 80)
        for turn_idx, turn in enumerate(sample_conv, 1):
            role_str = turn.get("role", "user").upper()
            content_str = turn.get("content", "")
            logger.info(f"  [Turn {turn_idx} - {role_str}]:")
            for line in content_str.strip().splitlines():
                logger.info(f"    {line}")
        logger.info("-" * 80)
        logger.info("APPLIED GEMMA 3 CHAT TEMPLATE (Pre-Tokenization Format):")
        logger.info("-" * 80)
        try:
            rendered_sample = tokenizer.apply_chat_template(sample_conv, tokenize=False)
        except Exception:
            rendered_sample = "<bos>"
            for m in sample_conv:
                role_val = "model" if m.get("role") in ("assistant", "model") else m.get("role", "user")
                rendered_sample += f"<start_of_turn>{role_val}\n{m.get('content', '')}<end_of_turn>\n"
        for line in rendered_sample.strip().splitlines():
            logger.info(f"    {line}")

        input_ids, labels = format_messages_with_prompt_mask(
            messages=sample_conv,
            tokenizer=tokenizer,
            max_seq_len=config.max_sequence_length,
            pad_to_max=False,
        )
        prompt_cnt = sum(1 for lbl in labels if lbl == -100)
        resp_cnt = sum(1 for lbl in labels if lbl != -100)
        logger.info("-" * 80)
        logger.info(
            f"Tokenization Check: {len(input_ids)} tokens total "
            f"({prompt_cnt} prompt-masked tokens with loss=-100, {resp_cnt} active assistant tokens)"
        )
        logger.info("=" * 80)

    logger.info(
        f"Tokenizing and indexing {len(conversations)} SFT conversation samples "
        f"(max_sequence_length={config.max_sequence_length})..."
    )
    sft_dataset = SFTDataset(
        conversations=conversations,
        tokenizer=tokenizer,
        max_seq_len=config.max_sequence_length,
    )
    if len(sft_dataset) == 0:
        raise ValueError(
            "SFTDataset contains 0 valid conversational samples after processing and filtering. Cannot train."
        )
    logger.info(f"Tokenization complete: {len(sft_dataset)} active conversational samples indexed.")

    dataloader = get_sft_dataloader(
        dataset=sft_dataset,
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
        warmup_ratio=config.warmup_ratio,
        total_steps=config.max_steps,
        eta_min_ratio=config.eta_min_ratio,
    )

    device_type = "cuda" if "cuda" in str(config.device) else ("mps" if "mps" in str(config.device) else "cpu")
    amp_dtype = (
        torch.bfloat16
        if config.amp_dtype == "bfloat16"
        else (torch.float16 if config.amp_dtype == "float16" else torch.float32)
    )

    scaler = None
    if config.amp_enabled and device_type == "cuda" and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda")  # type: ignore[attr-defined]

    model.train()
    step = 0
    accum_loss = 0.0
    accum_batches = 0
    t0 = time.time()
    step_start_time = time.time()

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
            tmp_checkpoint_path = f"{checkpoint_path}.tmp"
            torch.save(
                {
                    "step": step_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": current_loss,
                    "config": dataclasses.asdict(config),
                },
                tmp_checkpoint_path,
            )
            os.replace(tmp_checkpoint_path, checkpoint_path)
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

    warmup_steps = config.warmup_steps or int(config.max_steps * (config.warmup_ratio or 0.03))
    logger.info(
        f"Starting SFT optimization loop for {config.max_steps} steps "
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
                    loss = strategy.compute_loss(model, prepared_batch)
                    loss_scaled = loss / config.gradient_accumulation_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        f"NaN or Inf SFT loss detected at step {step + 1}. "
                        "Skipping backward pass and resetting accumulated gradients."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss = 0.0
                    accum_batches = 0
                    continue

                if scaler is not None:
                    scaler.scale(loss_scaled).backward()  # type: ignore[no-untyped-call]
                else:
                    loss_scaled.backward()  # type: ignore[no-untyped-call]

                accum_loss += loss.item()
                accum_batches += 1

                if accum_batches % config.gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        if not torch.isfinite(grad_norm):
                            logger.warning(
                                f"Non-finite gradient norm ({grad_norm}) detected at step {step + 1}. "
                                "Skipping optimizer step."
                            )
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            accum_loss = 0.0
                            accum_batches = 0
                            continue
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        if not torch.isfinite(grad_norm):
                            logger.warning(
                                f"Non-finite gradient norm ({grad_norm}) detected at step {step + 1}. "
                                "Skipping optimizer step."
                            )
                            optimizer.zero_grad(set_to_none=True)
                            accum_loss = 0.0
                            accum_batches = 0
                            continue
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
                        phase = f"Warmup ({step}/{warmup_steps})" if step <= warmup_steps else "Training"
                        logger.info(
                            f"Step {step}/{config.max_steps} [{phase}] | "
                            f"Loss: {step_loss:.4f} | LR: {current_lr:.2e} | "
                            f"Step Time: {step_time:.2f}s | Total Elapsed: {elapsed:.1f}s"
                        )
                    step_start_time = time.time()

                    # Combined eval and save interval
                    if step % config.eval_interval == 0 or step == config.max_steps:
                        save_if_better(step, step_loss)
    finally:
        logger.info(f"SFT training completed. Top {len(best_checkpoints)} checkpoints retained in {config.output_dir}")


if __name__ == "__main__":
    main()
