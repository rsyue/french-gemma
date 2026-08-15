"""
Dynamic CLI argument parser and TrainingConfig plumbing for train package.
"""

import argparse
import dataclasses
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

from src.config import TrainingConfig, parse_data_mix_config


def str_to_bool(val: Any) -> bool:
    """Convert string representations to boolean values."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        val_lower = val.lower()
        if val_lower in ("yes", "true", "t", "y", "1"):
            return True
        elif val_lower in ("no", "false", "f", "n", "0"):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{val}'.")


def str_or_int(val: Any) -> Union[int, str]:
    """Convert integer strings to int, otherwise return string."""
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        if val.isdigit():
            return int(val)
        return val
    return str(val)


def parse_args_to_config(args_list: Optional[List[str]] = None) -> TrainingConfig:
    """
    Parses CLI overrides dynamically based on TrainingConfig fields,
    merges them with YAML configuration values, and uses defaults if not present.
    """
    parser = argparse.ArgumentParser(description="French Gemma 3 Pretraining & Training CLI Launcher")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (e.g., configs/mlx_config.yaml)",
    )
    parser.add_argument(
        "--model",
        "--model_id",
        "--model-id",
        type=str,
        dest="model_id",
        default=None,
        help="HuggingFace model ID or local checkpoint path",
    )

    for f in dataclasses.fields(TrainingConfig):
        if f.name in ("freeze_schedule", "model_id"):
            continue

        dash_name = f"--{f.name.replace('_', '-')}"
        underscore_name = f"--{f.name}"

        field_type: Any
        if f.name == "data_mix":
            field_type = str
        else:
            field_type = f.type
            origin = get_origin(field_type)
            if origin is Union:
                args_of_union = get_args(field_type)
                non_none_types = [t for t in args_of_union if t is not type(None)]
                if str in non_none_types and int in non_none_types:
                    field_type = str_or_int
                elif non_none_types:
                    field_type = non_none_types[0]

        kwargs: Dict[str, Any] = {
            "type": field_type,
            "default": None,
            "dest": f.name,
            "help": f"Override {f.name} (default from config/dataclass)",
        }

        if field_type is bool:
            kwargs["type"] = str_to_bool

        if dash_name != underscore_name:
            parser.add_argument(dash_name, underscore_name, **kwargs)
        else:
            parser.add_argument(dash_name, **kwargs)

    parsed = parser.parse_args(args_list)

    if parsed.config:
        config = TrainingConfig.from_yaml(parsed.config)
    else:
        config = TrainingConfig()

    for f in dataclasses.fields(TrainingConfig):
        if f.name == "freeze_schedule":
            continue
        val = getattr(parsed, f.name, None)
        if val is not None:
            if f.name == "data_mix":
                config.data_mix = parse_data_mix_config(val)
            else:
                setattr(config, f.name, val)

    return config
