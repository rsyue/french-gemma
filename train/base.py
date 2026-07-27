"""
Base abstractions, interfaces, and union types for the train package.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable

import torch
import torch.nn as nn

from src.config import TrainingConfig


@runtime_checkable
class TrainingStrategyProtocol(Protocol):
    """Protocol defining the interface for all training strategies."""

    def compute_loss(self, model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
        ...


class AbstractTrainingStrategy(ABC):
    """Abstract Base Class for modular training strategies (Pretraining, RLHF, DPO, etc.)."""

    @abstractmethod
    def compute_loss(self, model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute training loss for a given batch."""
        pass

    def prepare_batch(self, batch: Dict[str, Any], device: str) -> Dict[str, Any]:
        """Move batch tensors to target device using non_blocking transfers."""
        prepared = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(device, non_blocking=True)
            else:
                prepared[k] = v
        return prepared


StrategyType = Union[AbstractTrainingStrategy, TrainingStrategyProtocol, str]


class BaseTrainer(ABC):
    """Abstract Base Class for dependency-injected trainers."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[TrainingConfig] = None,
        strategy: Optional[AbstractTrainingStrategy] = None,
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.strategy = strategy

    @abstractmethod
    def setup(self) -> None:
        """Initialize optimizers, schedulers, dataloaders, and hardware state."""
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single training optimization step."""
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Execute evaluation loop and return metric dictionary."""
        pass

    @abstractmethod
    def save_checkpoint(self, step: int, metrics: Dict[str, float]) -> str:
        """Save model and optimizer checkpoint."""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Clean up process groups, logging handlers, and memory."""
        pass
