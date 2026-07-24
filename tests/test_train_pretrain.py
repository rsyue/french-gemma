"""
Unit tests for train.cli and train.pretrain CLI entrypoint.
"""

from src.config import TrainingConfig
from train.cli import parse_args_to_config


def test_parse_args_overrides():
    """Verify CLI arguments override default TrainingConfig and YAML values."""
    cli_args = [
        "--config", "configs/mlx_config.yaml",
        "--model", "google/gemma-3-270m-it",
        "--device", "cpu",
        "--batch_size", "4",
        "--learning_rate", "2e-4",
    ]

    config = parse_args_to_config(cli_args)
    assert isinstance(config, TrainingConfig)
    assert config.device == "cpu"
    assert config.batch_size == 4
    assert config.learning_rate == 2e-4
    assert config.model_id == "google/gemma-3-270m-it"


def test_parse_args_freeze_schedule_empty_by_default():
    """Verify default freeze_schedule is empty for pretrain runs."""
    config = parse_args_to_config([])
    assert config.freeze_schedule == {}


def test_parse_args_num_examples_all():
    """Verify --num-examples all and numeric strings are accepted by the CLI parser."""
    config = parse_args_to_config(["--num-examples", "all"])
    assert config.num_examples == "all"

    config_num = parse_args_to_config(["--num_examples", "500"])
    assert config_num.num_examples == 500
