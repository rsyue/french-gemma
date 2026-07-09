"""
Unit Tests for Distributed Data Parallel (DDP) Guards and Dataset Chunk Injection.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import torch

from src.dataset import PackedTextDataset, get_dataloader
from src.trainer import Pretrainer


def test_packed_text_dataset_chunks_injection():
    # Test that providing pre-computed chunks bypasses tokenization/packing logic
    custom_chunks = [[1, 2, 3], [4, 5, 6]]
    dataset = PackedTextDataset(
        texts=None,
        tokenizer=None,
        max_seq_len=3,
        stride=1,
        chunks=custom_chunks,
    )
    assert dataset.chunks == custom_chunks
    assert len(dataset) == 2
    assert dataset[0] == {"input_ids": [1, 2, 3]}


def test_get_dataloader_with_sampler():
    custom_chunks = [[1, 2, 3]] * 10
    dataset = PackedTextDataset(max_seq_len=3, chunks=custom_chunks)
    
    # Mock sampler
    mock_sampler = MagicMock()
    
    # When a sampler is provided, shuffle should be disabled or handled by the sampler
    dl = get_dataloader(
        dataset=dataset,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        sampler=mock_sampler,
    )
    
    assert dl.sampler is mock_sampler


@patch("torch.distributed.is_available", return_value=True)
@patch("torch.distributed.is_initialized", return_value=True)
@patch("torch.distributed.get_rank", return_value=1)  # Worker process (Rank 1)
def test_pretrainer_worker_process_guards(mock_get_rank, mock_is_initialized, mock_is_available):
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def state_dict(self):
            return {"weights": torch.ones(10)}

    class MockOptimizer:
        def state_dict(self):
            return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Instantiate trainer on Rank 1 (Worker process)
        trainer = Pretrainer(
            model=MockModel(),
            tokenizer=None,
            train_dataloader=None,
            val_dataloader=None,
            optimizer=MockOptimizer(),
            lr_scheduler=None,
            freeze_manager=None,
            device="cpu",
            amp_enabled=False,
            output_dir=tmpdir,
            tb_log_dir=os.path.join(tmpdir, "runs"),
        )
        
        # Guard assertions
        assert not trainer.is_main_process
        assert trainer.writer is None  # Worker should not initialize SummaryWriter
        
        # Test that checkpoint saving methods do not write any files on rank 1
        trainer.save_checkpoint(global_step=100)
        trainer.save_best_perplexity_checkpoint(global_step=1, perplexity=50.0)
        trainer.save_best_loss_checkpoint(global_step=1, train_loss=1.0)
        
        # Verify no files were created in output_dir
        assert len(os.listdir(tmpdir)) == 0
        trainer.close()


@patch("torch.distributed.is_available", return_value=True)
@patch("torch.distributed.is_initialized", return_value=True)
@patch("torch.distributed.get_rank", return_value=0)  # Main process (Rank 0)
def test_pretrainer_main_process_ddp(mock_get_rank, mock_is_initialized, mock_is_available):
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def state_dict(self):
            return {"weights": torch.ones(10)}

    class MockOptimizer:
        def state_dict(self):
            return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Pretrainer(
            model=MockModel(),
            tokenizer=None,
            train_dataloader=None,
            val_dataloader=None,
            optimizer=MockOptimizer(),
            lr_scheduler=None,
            freeze_manager=None,
            device="cpu",
            amp_enabled=False,
            output_dir=tmpdir,
            tb_log_dir=os.path.join(tmpdir, "runs"),
        )
        
        # Main process assertions
        assert trainer.is_main_process
        assert trainer.writer is not None
        
        # Verify checkpoint saving works on rank 0
        trainer.save_checkpoint(global_step=100)
        assert len(os.listdir(tmpdir)) > 0
        trainer.close()
