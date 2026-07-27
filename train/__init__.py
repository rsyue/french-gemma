"""
Train Package: Modular, extensible training framework for French Gemma 3.
"""

from train.base import AbstractTrainingStrategy, BaseTrainer, StrategyType
from train.builder import ModularTrainer, TrainingFactory
from train.strategies.pretrain import PretrainStrategy

__all__ = [
    "BaseTrainer",
    "AbstractTrainingStrategy",
    "StrategyType",
    "PretrainStrategy",
    "TrainingFactory",
    "ModularTrainer",
]
