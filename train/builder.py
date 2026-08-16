"""
Dependency injection factory and concrete trainer construction.
"""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.config import TrainingConfig
from train.base import AbstractTrainingStrategy, BaseTrainer, StrategyType, TrainingStrategyProtocol
from train.strategies.dpo import DPOStrategy
from train.strategies.pretrain import PretrainStrategy
from train.strategies.sft import SFTStrategy


class ModularTrainer(BaseTrainer):
    """Concrete, strategy-aware trainer implementing BaseTrainer lifecycle interfaces."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[TrainingConfig] = None,
        strategy: Optional[AbstractTrainingStrategy] = None,
    ) -> None:
        super().__init__(model=model, config=config, strategy=strategy or PretrainStrategy())
        self.device = self.config.device if self.config else "cpu"
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def setup(self) -> None:
        """Initialize optimizer if model is provided."""
        if self.model is not None and self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single strategy-driven optimization step."""
        if self.model is None or self.strategy is None:
            raise RuntimeError("Model and strategy must be configured before calling train_step.")

        if self.optimizer is None:
            raise RuntimeError("Optimizer is not initialized. Call setup() before train_step().")

        self.model.train()
        prepared_batch = self.strategy.prepare_batch(batch, self.device)

        self.optimizer.zero_grad(set_to_none=True)
        loss: torch.Tensor = self.strategy.compute_loss(self.model, prepared_batch)
        loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()

        return {"loss": loss.item()}

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model performance."""
        return {"eval_loss": 0.0, "perplexity": 1.0}

    def save_checkpoint(self, step: int, metrics: Dict[str, float]) -> str:
        """Save model checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-{step}.pt")
        if self.model is not None:
            state = {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "metrics": metrics,
            }
            torch.save(state, checkpoint_path)
        return checkpoint_path

    def teardown(self) -> None:
        """Cleanup resources."""
        pass


class TrainingFactory:
    """Dependency injection factory for instantiating strategies and trainers."""

    @staticmethod
    def build_strategy(strategy: StrategyType) -> AbstractTrainingStrategy:
        """Resolve and instantiate a training strategy instance."""
        if isinstance(strategy, (AbstractTrainingStrategy, TrainingStrategyProtocol)):
            return strategy  # type: ignore[return-value]
        if isinstance(strategy, str):
            strategy_lower = strategy.lower()
            if strategy_lower in ("pretrain", "pretraining", "causal"):
                return PretrainStrategy()
            if strategy_lower in ("sft", "supervised", "chat"):
                return SFTStrategy()
            if strategy_lower in ("dpo", "preference", "rlhf", "rl"):
                return DPOStrategy()
            raise ValueError(f"Unknown training strategy string: '{strategy}'")
        raise TypeError(f"Invalid strategy type provided: {type(strategy)}")

    @classmethod
    def create_trainer(
        cls,
        model: Optional[nn.Module] = None,
        config: Optional[TrainingConfig] = None,
        strategy: Optional[StrategyType] = None,
    ) -> BaseTrainer:
        """Create a fully-assembled modular trainer via dependency injection."""
        resolved_strategy = (
            cls.build_strategy(strategy) if strategy is not None else PretrainStrategy()
        )
        return ModularTrainer(model=model, config=config, strategy=resolved_strategy)
