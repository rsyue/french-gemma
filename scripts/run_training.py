"""
Pretraining Script for French Gemma 3 supporting Single-GPU, MPS, and Multi-GPU (DDP) runs.
"""

import argparse
import dataclasses
import logging
import os
from typing import Any, List, Optional, Union, get_args, get_origin

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerFast

from src.config import TrainingConfig
from src.dataset import PackedTextDataset, get_dataloader, load_french_dataset, train_custom_tokenizer
from src.model import FrenchGemmaModel
from src.scheduler import FreezeManager, get_cosine_warmup_scheduler
from src.trainer import Pretrainer

logger = logging.getLogger(__name__)


def parse_and_load_config(args_list: Optional[List[str]] = None) -> TrainingConfig:
    """
    Parses CLI overrides dynamically based on TrainingConfig fields,
    merges them with YAML configuration values, and uses defaults if not present.
    """
    parser = argparse.ArgumentParser(description="Pretrain French Gemma 3")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mlx_config.yaml",
        help="Path to YAML configuration file",
    )

    # Dynamically add all fields of TrainingConfig as command line arguments
    for f in dataclasses.fields(TrainingConfig):
        if f.name == "freeze_schedule":
            continue

        arg_name = f"--{f.name.replace('_', '-')}"
        
        # Resolve field type for Optionals
        field_type = f.type
        origin = get_origin(field_type)
        if origin is Union:
            args_of_union = get_args(field_type)
            non_none_types = [t for t in args_of_union if t is not type(None)]
            if non_none_types:
                field_type = non_none_types[0]

        kwargs: dict[str, Any] = {
            "type": field_type,
            "default": None,
            "help": f"Override {f.name} (default from config/dataclass)",
        }

        # Specially handle bool type
        if field_type is bool:
            def str_to_bool(val: str) -> bool:
                if isinstance(val, bool):
                    return val
                if val.lower() in ('yes', 'true', 't', 'y', '1'):
                    return True
                elif val.lower() in ('no', 'false', 'f', 'n', '0'):
                    return False
                raise argparse.ArgumentTypeError('Boolean value expected.')
            kwargs["type"] = str_to_bool

        parser.add_argument(arg_name, **kwargs)

    # Parse arguments
    args = parser.parse_args(args_list)

    # Load from config file if exists
    if args.config and os.path.exists(args.config):
        config = TrainingConfig.from_yaml(args.config)
    else:
        logger.warning(
            f"Configuration file '{args.config}' not found or not specified. "
            "Using default TrainingConfig."
        )
        config = TrainingConfig()

    # Override config with any CLI arguments explicitly passed (i.e. not None)
    for f in dataclasses.fields(TrainingConfig):
        if f.name == "freeze_schedule":
            continue
        val = getattr(args, f.name, None)
        if val is not None:
            setattr(config, f.name, val)

    return config


def main() -> None:
    # 1. Parse configuration and overrides
    config = parse_and_load_config()

    # 2. Check if running under torchrun/DDP
    is_distributed = "WORLD_SIZE" in os.environ
    if is_distributed:
        # DDP process group initialization
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = "cpu"  # Will be overridden by config device

    # Configure logging (only Rank 0 logs INFO; others log WARNING)
    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
        level=log_level,
    )

    if not is_distributed:
        device = config.device
    logger.info(f"Loaded config. Device target: {device} | Distributed: {is_distributed}")


    # Set directories and cache paths
    tokenizer_dir = os.path.join(config.data_cache_dir, "tokenizer_checkpoint")
    cache_path = os.path.join(config.data_cache_dir, "packed_dataset_cache.bin")

    # 4. Data preparation only on Rank 0
    if rank == 0:
        logger.info("Main process (Rank 0) starting data preparation...")
        os.makedirs(config.data_cache_dir, exist_ok=True)
        if os.path.exists(cache_path):
            logger.warning(
                f"Binary cache file already exists at {cache_path}. "
                "Overwriting the existing cache. Ensure this is intentional "
                "and no other runs are sharing this directory."
            )

        # Load dataset
        texts = load_french_dataset(
            dataset_path=config.dataset_path,
            dataset_name=config.dataset_name,
            split="train[:100]",
        )
        logger.info(f"Loaded {len(texts)} articles/paragraphs.")

        # Train a custom tokenizer
        logger.info("Training custom ByteLevelBPETokenizer on main process...")
        tokenizer = train_custom_tokenizer(texts, vocab_size=1000, save_dir=tokenizer_dir)

        # Create packed sequences by streaming tokenization directly to the binary cache file on disk
        logger.info("Packing tokens and generating binary cache file on main process...")
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

        with open(cache_path, "wb") as f:
            for text in texts:
                if not text.strip():
                    continue
                tokens = tokenizer.encode(text, add_special_tokens=False)
                doc_ids = []
                if bos_id is not None:
                    doc_ids.append(bos_id)
                doc_ids.extend(tokens)
                if eos_id is not None:
                    doc_ids.append(eos_id)
                
                arr = np.array(doc_ids, dtype=np.uint32)
                f.write(arr.tobytes())
        logger.info(f"Main process finished data prep. Cached packed binary to {cache_path}")

    # 5. Barrier synchronization for workers
    if is_distributed:
        logger.info(f"Rank {rank} waiting for main process to complete data prep...")
        dist.barrier()
        logger.info(f"Rank {rank} released from barrier.")

    # 6. Load tokenizer & dataset on all ranks
    logger.info(f"Rank {rank} loading tokenizer from {tokenizer_dir}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    logger.info(f"Rank {rank} loading packed dataset from binary cache: {cache_path}")
    dataset = PackedTextDataset(
        bin_path=cache_path,
        tokenizer=tokenizer,
        max_seq_len=config.max_sequence_length,
        stride=50,
    )
    logger.info(f"Rank {rank} instantiated dataset with {len(dataset)} sequences.")

    # 7. Configure Distributed Samplers & Dataloaders
    train_sampler: Optional[DistributedSampler[int]] = None
    val_sampler: Optional[DistributedSampler[int]] = None
    if is_distributed:
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
        val_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = get_dataloader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    val_dataloader = get_dataloader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
        shuffle=False,
        sampler=val_sampler,
    )

    # 8. Initialize FrenchGemmaModel
    logger.info(f"Rank {rank} instantiating FrenchGemmaModel on device {device}...")
    model: torch.nn.Module = FrenchGemmaModel(
        model_id=config.model_id,
        vocab_size=len(tokenizer),
        embedding_noise_std=config.embedding_noise_std,
    ).to(device)

    # Wrap model in DistributedDataParallel if distributed
    if is_distributed:
        device_ids = [local_rank] if "cuda" in device else None
        output_device = local_rank if "cuda" in device else None
        model = DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=True,
        )

    # Compile the model if compile option is enabled
    if config.compile:
        logger.info("Compiling model (torch.compile)...")
        model = torch.compile(model)  # type: ignore[assignment]

    # 9. Configure Optimizer, Scheduler, and FreezeManager
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps=config.warmup_steps, T_0=1000)
    
    # Extract raw model for FreezeManager
    raw_model: torch.nn.Module = model
    if config.compile:
        raw_model = getattr(model, "_orig_mod", model)
    if isinstance(raw_model, DistributedDataParallel):
        raw_model = raw_model.module
    freeze_manager = FreezeManager(raw_model, config.freeze_schedule)

    # 10. Instantiate Pretrainer
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
    )

    # 11. Run pretraining loop (mocking 3 epochs for training run example)
    logger.info(f"Rank {rank} starting pretraining loop...")
    global_step = 0
    for epoch in range(3):
        if global_step >= config.max_steps:
            logger.info(f"Reached max steps: {global_step} >= {config.max_steps}. Stopping pretraining.")
            break
        logger.info(f"--- Starting Epoch {epoch} ---")
        global_step = trainer.train_epoch(epoch=epoch, global_step=global_step)

    logger.info(f"Rank {rank} pretraining run completed successfully!")
    trainer.close()

    # 12. Cleanup process group
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
