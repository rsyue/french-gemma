import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    warmup_steps: int = 100
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
    vocab_size: Optional[int] = None
    output_dir: str = "./checkpoints"
    tb_log_dir: str = "./runs"
    embedding_noise_std: float = 0.0

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Parse freeze schedule keys as integers
        if "freeze_schedule" in data and data["freeze_schedule"] is not None:
            data["freeze_schedule"] = {int(k): list(v) for k, v in data["freeze_schedule"].items()}
            
        return cls(**data)
