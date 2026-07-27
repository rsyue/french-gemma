"""
Unit tests for the modular train package (train.base, train.strategies, train.builder).
"""

import os
import tempfile
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from src.config import TrainingConfig
from train.base import AbstractTrainingStrategy, BaseTrainer
from train.builder import ModularTrainer, TrainingFactory
from train.strategies.pretrain import PretrainStrategy


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, input_ids=None, labels=None, **kwargs):
        device = input_ids.device if input_ids is not None else torch.device("cpu")
        logits = self.linear(torch.randn(input_ids.shape[0], input_ids.shape[1], 10, device=device))
        loss = None
        if labels is not None:
            loss = self.linear(torch.randn(1, 10, device=device)).sum()
        return type("Output", (), {"loss": loss, "logits": logits})()


class DummyStrategy(AbstractTrainingStrategy):
    def compute_loss(self, model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
        outputs = model(**batch)
        return outputs.loss


def test_base_trainer_abstract():
    """Verify BaseTrainer cannot be instantiated directly without implementing abstract methods."""
    with pytest.raises(TypeError):
        BaseTrainer()  # type: ignore[abstract]


def test_pretrain_strategy():
    """Verify PretrainStrategy computes loss and handles batch forwarding."""
    strategy = PretrainStrategy()
    model = DummyModel()
    batch = {
        "input_ids": torch.randint(0, 10, (2, 16)),
        "labels": torch.randint(0, 10, (2, 16)),
    }
    loss = strategy.compute_loss(model, batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_training_factory_build_strategy():
    """Verify TrainingFactory builds appropriate strategies based on type string or instance."""
    strategy = TrainingFactory.build_strategy("pretrain")
    assert isinstance(strategy, PretrainStrategy)

    custom_strategy = DummyStrategy()
    injected_strategy = TrainingFactory.build_strategy(custom_strategy)
    assert injected_strategy is custom_strategy


def test_training_factory_create_trainer():
    """Verify TrainingFactory builds a functional trainer instance via dependency injection."""
    config = TrainingConfig(device="cpu", max_steps=10)
    model = DummyModel()
    strategy = PretrainStrategy()

    trainer = TrainingFactory.create_trainer(
        model=model,
        config=config,
        strategy=strategy,
    )
    assert isinstance(trainer, BaseTrainer)
    assert trainer.strategy is strategy


def test_modular_trainer_lifecycle():
    """Verify ModularTrainer setup, train_step, and save_checkpoint lifecycle methods."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(device="cpu", output_dir=tmpdir)
        model = DummyModel()
        trainer = ModularTrainer(model=model, config=config)

        # Uninitialized optimizer raises error on train_step
        with pytest.raises(RuntimeError):
            trainer.train_step({"input_ids": torch.randint(0, 10, (2, 16))})

        trainer.setup()
        step_result = trainer.train_step({
            "input_ids": torch.randint(0, 10, (2, 16)),
            "labels": torch.randint(0, 10, (2, 16)),
        })
        assert "loss" in step_result
        assert isinstance(step_result["loss"], float)

        checkpoint_path = trainer.save_checkpoint(step=1, metrics={"loss": step_result["loss"]})
        assert os.path.exists(checkpoint_path)
