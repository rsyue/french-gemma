"""
Pretraining Entrypoint Module for French Gemma 3.
Executable via: python -m train.pretrain --config configs/mlx_config.yaml --model google/gemma-3-270m-it
"""

import logging
import os
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerFast

from src.dataset import (
    PackedTextDataset,
    get_dataloader,
    is_data_prepared,
    load_dataset_mix,
    load_french_dataset,
    prepare_and_pack_data,
    train_custom_tokenizer,
    wait_for_data_prep,
)
from src.model import FrenchGemmaModel
from src.scheduler import FreezeManager, get_cosine_warmup_scheduler
from src.trainer import Pretrainer
from train.builder import TrainingFactory
from train.cli import parse_args_to_config

logger = logging.getLogger(__name__)


def main() -> None:
    """Main CLI entrypoint for French Gemma 3 pretraining."""
    config = parse_args_to_config()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    is_distributed = "WORLD_SIZE" in os.environ
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = config.device

    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
        level=log_level,
    )

    logger.info(
        f"Pretraining launched. Model: {config.model_id} | Device: {device} | Distributed: {is_distributed}"
    )

    strategy = TrainingFactory.build_strategy("pretrain")
    logger.info(f"Loaded strategy: {type(strategy).__name__}")

    tokenizer_dir = os.path.join(config.data_cache_dir, "tokenizer_checkpoint")
    cache_path = os.path.join(config.data_cache_dir, "packed_dataset_cache.bin")

    if rank == 0:
        os.makedirs(config.data_cache_dir, exist_ok=True)
        if not is_data_prepared(cache_path, tokenizer_dir):
            if config.data_mix:
                logger.info(
                    f"Loading French dataset mix with {len(config.data_mix)} sources "
                    f"(total_examples={config.num_examples})..."
                )
                texts = load_dataset_mix(
                    data_mix=config.data_mix,
                    total_examples=config.num_examples,
                    seed=config.seed,
                )
            else:
                num_ex_str = str(config.num_examples).strip().lower()
                if num_ex_str in ("all", "full", "none", "0"):
                    dataset_split = "train"
                elif num_ex_str.isdigit():
                    dataset_split = f"train[:{num_ex_str}]"
                else:
                    dataset_split = str(config.num_examples)

                logger.info(f"Loading French dataset split: '{dataset_split}'...")
                texts = load_french_dataset(
                    dataset_path=config.dataset_path,
                    dataset_name=config.dataset_name,
                    split=dataset_split,
                )
            vocab_size = config.vocab_size or 35000
            logger.info(f"Training custom tokenizer with vocab_size={vocab_size}...")
            tokenizer = train_custom_tokenizer(texts, vocab_size=vocab_size, save_dir=tokenizer_dir)

            prepare_and_pack_data(
                texts=texts,
                tokenizer=tokenizer,
                cache_path=cache_path,
                packing_batch_size=config.packing_batch_size,
                packing_log_interval=config.packing_log_interval,
            )
        else:
            logger.info("Dataset and tokenizer already prepared. Skipping preparation step.")
    elif is_distributed:
        wait_for_data_prep(cache_path, tokenizer_dir, timeout=config.dist_timeout_seconds)

    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=config.dist_timeout_seconds),
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        dist.barrier()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    dataset = PackedTextDataset(
        bin_path=cache_path,
        tokenizer=tokenizer,
        max_seq_len=config.max_sequence_length,
        stride=50,
    )

    train_sampler: Optional[DistributedSampler[int]] = None
    val_sampler: Optional[DistributedSampler[int]] = None
    if is_distributed:
        train_sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed
        )
        val_sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

    train_dataloader = get_dataloader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        seed=config.seed,
    )
    val_dataloader = get_dataloader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
        shuffle=False,
        sampler=val_sampler,
        seed=config.seed,
    )

    model: torch.nn.Module = FrenchGemmaModel(
        model_id=config.model_id,
        vocab_size=len(tokenizer),
        embedding_noise_std=config.embedding_noise_std,
    ).to(device)

    if is_distributed:
        device_ids = [local_rank] if "cuda" in device else None
        output_device = local_rank if "cuda" in device else None
        model = DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=True,
        )

    if config.compile:
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    lr_scheduler = get_cosine_warmup_scheduler(
        optimizer,
        total_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        warmup_ratio=config.warmup_ratio,
        T_0=config.T_0,
        T_mult=config.T_mult,
        num_cycles=config.num_cycles,
        eta_min_ratio=config.eta_min_ratio,
    )

    raw_model: torch.nn.Module = model
    if config.compile:
        raw_model = getattr(model, "_orig_mod", model)
    if isinstance(raw_model, DistributedDataParallel):
        raw_model = raw_model.module

    freeze_manager = FreezeManager(raw_model, config.freeze_schedule)

    trainer = Pretrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        freeze_manager=freeze_manager,
        device=device,
        amp_enabled=config.amp_enabled,
        amp_dtype=config.amp_dtype,
        grad_accum_steps=config.gradient_accumulation_steps,
        log_interval=config.log_interval,
        eval_interval=config.eval_interval,
        save_interval=config.save_interval,
        output_dir=config.output_dir,
        tb_log_dir=config.tb_log_dir,
        max_eval_batches=config.max_eval_batches,
        max_checkpoints=config.max_checkpoints,
        max_steps=config.max_steps,
        repetition_penalty=config.repetition_penalty,
    )

    try:
        global_step = 0
        epoch = 0
        while global_step < config.max_steps:
            global_step = trainer.train_epoch(epoch=epoch, global_step=global_step)
            epoch += 1
    finally:
        trainer.close()
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
