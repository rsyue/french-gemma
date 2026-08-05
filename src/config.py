"""
Configuration Loader and Settings.

This module parses YAML files into the TrainingConfig dataclass, defining training,
hardware execution, datasets, tokenizer, optimization, and checkpoint settings.
"""

import inspect
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import yaml


@dataclass
class TrainingConfig:
    model_id: str = "google/gemma-3-270m-it"
    dataset_path: str = "wikimedia/wikipedia"
    dataset_name: str = "20231101.fr"
    device: str = "cpu"
    max_sequence_length: int = 512
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_steps: Optional[int] = None
    warmup_ratio: float = 0.03
    T_0: Optional[int] = None
    T_mult: int = 2
    num_cycles: int = 1
    eta_min_ratio: float = 0.01
    max_steps: int = 5000

    compile: bool = False
    amp_enabled: bool = True
    amp_dtype: str = "bfloat16"
    log_interval: int = 50
    save_interval: int = 1000
    eval_interval: int = 500
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    freeze_schedule: Dict[int, List[int]] = field(default_factory=dict)
    vocab_size: int = 35000
    num_examples: Union[int, str] = "all"
    output_dir: str = "./checkpoints"
    data_cache_dir: str = "./data_cache"
    tb_log_dir: str = "./runs"
    embedding_noise_std: float = 0.0
    max_eval_batches: int = 20
    max_checkpoints: int = 5
    repetition_penalty: float = 1.2
    packing_batch_size: int = 10000
    packing_log_interval: int = 10
    seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            data = {}

        if "freeze_schedule" in data and data["freeze_schedule"] is not None:
            data["freeze_schedule"] = {int(k): list(v) for k, v in data["freeze_schedule"].items()}

        valid_keys = inspect.signature(cls).parameters.keys()
        filtered_data = {
            k: v for k, v in data.items()
            if k in valid_keys and v is not None
        }

        return cls(**filtered_data)
