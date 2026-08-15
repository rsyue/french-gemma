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

from src.dataset import (
    PackedTextDataset,
    get_dataloader,
    is_data_prepared,
    prepare_and_pack_data,
    train_custom_tokenizer,
    wait_for_data_prep,
)
from src.model import FrenchGemmaModel
from src.scheduler import FreezeManager, get_cosine_warmup_scheduler
from src.trainer import Pretrainer


def ddp_worker(rank: int, world_size: int, tmpdir: str, mock_texts: list[str]) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    try:
        tokenizer_dir = os.path.join(tmpdir, "tokenizer")
        cache_path = os.path.join(tmpdir, "dataset.bin")

        if rank == 0:
            tokenizer = train_custom_tokenizer(mock_texts, vocab_size=100, save_dir=tokenizer_dir)
            prepare_and_pack_data(
                texts=mock_texts,
                tokenizer=tokenizer,
                cache_path=cache_path,
                packing_batch_size=10,
            )
        else:
            wait_for_data_prep(cache_path, tokenizer_dir, timeout=10)

        assert is_data_prepared(cache_path, tokenizer_dir)

        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        dist.barrier()

        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        dataset = PackedTextDataset(
            bin_path=cache_path,
            tokenizer=tokenizer,
            max_seq_len=8,
            stride=2,
        )

        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
        dataloader = get_dataloader(
            dataset=dataset,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
            sampler=train_sampler,
        )

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

        model = DistributedDataParallel(model, find_unused_parameters=True)

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

        global_step = trainer.train_epoch(epoch=0, global_step=0)
        assert global_step > 0

        chkpts_dir = os.path.join(tmpdir, "checkpoints")
        if rank == 0:
            assert len(os.listdir(chkpts_dir)) > 0

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
        mp.spawn(
            ddp_worker,
            args=(world_size, tmpdir, mock_texts),
            nprocs=world_size,
            join=True,
        )

