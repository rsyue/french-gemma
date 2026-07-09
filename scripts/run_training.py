"""
Pretraining Script for French Gemma 3 supporting Single-GPU, MPS, and Multi-GPU (DDP) runs.
"""

import argparse
import logging
import os
from typing import Optional

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


def main() -> None:
    # 1. Parse command line arguments
    parser = argparse.ArgumentParser(description="Pretrain French Gemma 3")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mlx_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data-cache-dir",
        type=str,
        default=None,
        help="Path to directory for caching tokenized data binary",
    )
    args = parser.parse_args()

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

    # 3. Load configuration
    config = TrainingConfig.from_yaml(args.config)
    if args.data_cache_dir is not None:
        config.data_cache_dir = args.data_cache_dir
    if not is_distributed:
        device = config.device
    logger.info(f"Loaded config from {args.config}. Device target: {device} | Distributed: {is_distributed}")

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

    # 9. Configure Optimizer, Scheduler, and FreezeManager
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps=config.warmup_steps, T_0=1000)
    
    # Extract raw model for FreezeManager
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
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
        output_dir=config.output_dir,
        tb_log_dir=config.tb_log_dir,
        log_interval=10,
        eval_interval=50,
        save_interval=100,
    )

    # 11. Run pretraining loop (mocking 3 epochs for training run example)
    logger.info(f"Rank {rank} starting pretraining loop...")
    global_step = 0
    for epoch in range(3):
        logger.info(f"--- Starting Epoch {epoch} ---")
        global_step = trainer.train_epoch(epoch=epoch, global_step=global_step)

    logger.info(f"Rank {rank} pretraining run completed successfully!")
    trainer.close()

    # 12. Cleanup process group
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
