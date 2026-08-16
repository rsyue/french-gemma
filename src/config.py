"""
Configuration Loader and Settings.

This module parses YAML files into the TrainingConfig dataclass, defining training,
hardware execution, datasets, data mixes, tokenizer, optimization, and checkpoint settings.
"""

import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class DatasetMixEntry:
    """Represents a single dataset source within a multi-dataset mix with proportional weighting."""

    dataset_path: str
    dataset_name: Optional[str] = None
    percentage: float = 100.0
    split: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMixEntry":
        path = data.get("dataset_path") or data.get("path")
        if not path:
            raise ValueError("DatasetMixEntry must specify 'dataset_path' or 'path'.")
        name = data.get("dataset_name") or data.get("name")
        raw_pct = None
        for key in ("percentage", "weight", "pct"):
            if key in data and data[key] is not None:
                raw_pct = data[key]
                break
        pct = float(raw_pct) if raw_pct is not None else 100.0
        split = data.get("split")
        return cls(
            dataset_path=str(path),
            dataset_name=str(name) if name is not None else None,
            percentage=pct,
            split=str(split) if split is not None else None,
        )


def parse_data_mix_config(raw_data_mix: Any) -> Optional[List[DatasetMixEntry]]:
    """
    Parses a raw data mix input (list of dictionaries, YAML file path, or serialized string)
    into a list of DatasetMixEntry instances.
    """
    if raw_data_mix is None:
        return None

    if isinstance(raw_data_mix, str):
        raw_str = raw_data_mix.strip()
        if os.path.exists(raw_str):
            with open(raw_str, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict) and "data_mix" in loaded:
                    raw_data_mix = loaded["data_mix"]
                elif isinstance(loaded, list):
                    raw_data_mix = loaded
                else:
                    raise ValueError(f"Invalid data mix YAML structure in file: {raw_str}")
        else:
            try:
                loaded = yaml.safe_load(raw_str)
                if isinstance(loaded, list):
                    raw_data_mix = loaded
                elif isinstance(loaded, dict) and "data_mix" in loaded:
                    raw_data_mix = loaded["data_mix"]
            except Exception:
                raise ValueError(f"Unable to parse data_mix string or locate YAML file: '{raw_str}'")

    if not isinstance(raw_data_mix, list):
        raise ValueError(f"Expected data_mix to be a list, got {type(raw_data_mix).__name__}")

    parsed_entries: List[DatasetMixEntry] = []
    for item in raw_data_mix:
        if isinstance(item, DatasetMixEntry):
            parsed_entries.append(item)
        elif isinstance(item, dict):
            parsed_entries.append(DatasetMixEntry.from_dict(item))
        else:
            raise ValueError(f"Unsupported data mix item type: {type(item).__name__}")

    return parsed_entries


@dataclass
class TrainingConfig:
    model_id: str = "google/gemma-3-270m-it"
    dataset_path: str = "wikimedia/wikipedia"
    dataset_name: str = "20231101.fr"
    data_mix: Optional[List[DatasetMixEntry]] = None
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
    save_interval: int = 500
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
    dist_timeout_seconds: int = 7200
    seed: int = 42
    dpo_beta: float = 0.1
    dpo_label_smoothing: float = 0.0
    ref_model_id: Optional[str] = None
    save_dir: Optional[str] = None
    pretrained_model_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.save_dir is not None:
            self.output_dir = self.save_dir
        else:
            self.save_dir = self.output_dir

        if not self.output_dir or not self.output_dir.strip():
            raise ValueError("output_dir / save_dir must be a non-empty path string.")
        if self.max_checkpoints < 1:
            raise ValueError(f"max_checkpoints must be >= 1, got {self.max_checkpoints}")
        if self.eval_interval < 1:
            raise ValueError(f"eval_interval must be >= 1, got {self.eval_interval}")
        if self.log_interval < 1:
            raise ValueError(f"log_interval must be >= 1, got {self.log_interval}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            data = {}

        if "save_dir" in data and ("output_dir" not in data or data["output_dir"] is None):
            data["output_dir"] = data["save_dir"]
        elif "output_dir" in data and ("save_dir" not in data or data["save_dir"] is None):
            data["save_dir"] = data["output_dir"]

        if "freeze_schedule" in data and data["freeze_schedule"] is not None:
            data["freeze_schedule"] = {int(k): list(v) for k, v in data["freeze_schedule"].items()}

        if "data_mix" in data and data["data_mix"] is not None:
            data["data_mix"] = parse_data_mix_config(data["data_mix"])

        valid_keys = inspect.signature(cls).parameters.keys()
        filtered_data = {
            k: v for k, v in data.items()
            if k in valid_keys and v is not None
        }

        return cls(**filtered_data)
