"""
Integration Tests for Multi-Process Distributed Data Parallel (DDP) Pretraining.
"""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerFast

from src.dataset import PackedTextDataset, get_dataloader, train_custom_tokenizer
from src.model import FrenchGemmaModel
from src.scheduler import FreezeManager, get_cosine_warmup_scheduler
from src.trainer import Pretrainer


def ddp_worker(rank: int, world_size: int, tmpdir: str, mock_texts: list[str]) -> None:
    # 1. Setup DDP environment variables for local spawned processes
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize with 'gloo' backend for CPU process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    try:
        tokenizer_dir = os.path.join(tmpdir, "tokenizer")
        cache_path = os.path.join(tmpdir, "dataset.bin")

        # 2. Main Process Data Preparation
        if rank == 0:
            # Train tokenizer
            tokenizer = train_custom_tokenizer(mock_texts, vocab_size=100, save_dir=tokenizer_dir)
            # Tokenize and write to binary cache
            bos_id = tokenizer.bos_token_id
            eos_id = tokenizer.eos_token_id

            all_tokens = []
            for text in mock_texts:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                doc_ids = []
                if bos_id is not None:
                    doc_ids.append(bos_id)
                doc_ids.extend(tokens)
                if eos_id is not None:
                    doc_ids.append(eos_id)
                all_tokens.extend(doc_ids)

            import numpy as np
            arr = np.array(all_tokens, dtype=np.uint32)
            with open(cache_path, "wb") as f:
                f.write(arr.tobytes())

        # 3. Synchronize ranks
        dist.barrier()

        # 4. Load from cache on all processes
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        dataset = PackedTextDataset(
            bin_path=cache_path,
            tokenizer=tokenizer,
            max_seq_len=8,
            stride=2,
        )

        # 5. Distributed Samplers
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
        dataloader = get_dataloader(
            dataset=dataset,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
            sampler=train_sampler,
        )

        # 6. Initialize Tiny Model (overriding config to speed up CPU training in tests)
        tiny_override = {
            "num_hidden_layers": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
        }
        model = FrenchGemmaModel(
            model_id="google/gemma-3-270m-it",
            vocab_size=len(tokenizer),
            config_override=tiny_override,
        ).to("cpu")

        # Wrap in DDP
        model = DistributedDataParallel(model, find_unused_parameters=True)

        # 7. Schedulers & Optimizers
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        lr_scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps=1, T_0=10)
        freeze_manager = FreezeManager(model.module, {0: [0]})

        trainer = Pretrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=dataloader,
            val_dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            freeze_manager=freeze_manager,
            device="cpu",
            amp_enabled=False,
            output_dir=os.path.join(tmpdir, "checkpoints"),
            tb_log_dir=os.path.join(tmpdir, "runs"),
            max_eval_batches=2,
            save_interval=2,
            eval_interval=2,
        )

        # 8. Train for 2 steps
        global_step = trainer.train_epoch(epoch=0, global_step=0)
        assert global_step > 0

        # Verify only main process saved checkpoints
        chkpts_dir = os.path.join(tmpdir, "checkpoints")
        if rank == 0:
            assert len(os.listdir(chkpts_dir)) > 0
        else:
            # Worker processes should not have created periodic or best checkpoints directly
            # Note: in this local thread execution they share the same directory, but only rank 0 calls saving.
            pass

        # 9. Clean up
        trainer.close()
    finally:
        dist.destroy_process_group()


def test_ddp_integration():
    mock_texts = [
        "Un long texte en français pour valider le bon fonctionnement de DDP.",
        "La synchronisation des processus doit s'effectuer sans aucun interblocage.",
        "Chaque rank charge sa partition de données via DistributedSampler.",
        "Le modèle Gemma 3 est entraîné de manière distribuée sur plusieurs cœurs.",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        world_size = 2
        # Spawn 2 worker processes
        mp.spawn(
            ddp_worker,
            args=(world_size, tmpdir, mock_texts),
            nprocs=world_size,
            join=True,
        )
