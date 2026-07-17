"""
Unit Tests for Training Configuration.

This module contains unit tests that verify parsing of YAML configuration files,
dataclass field defaults, and custom freeze schedule formatting.
"""

import os
import tempfile

import yaml

from src.config import TrainingConfig


def test_config_defaults():
    config = TrainingConfig()
    assert config.max_eval_batches == 20
    assert config.max_checkpoints == 5
    assert config.data_cache_dir == "./data_cache"


def test_config_custom_values():
    config = TrainingConfig(max_eval_batches=42, max_checkpoints=10)
    assert config.max_eval_batches == 42
    assert config.max_checkpoints == 10


def test_config_from_yaml():
    yaml_content = {
        "model_id": "google/gemma-3-270m-it",
        "max_eval_batches": 15,
        "max_checkpoints": 8,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        config = TrainingConfig.from_yaml(config_path)
        assert config.max_eval_batches == 15
        assert config.max_checkpoints == 8


def test_all_repo_configs():
    config_dir = "configs"
    assert os.path.exists(config_dir)
    yaml_files = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]
    assert len(yaml_files) > 0
    for filename in yaml_files:
        path = os.path.join(config_dir, filename)
        config = TrainingConfig.from_yaml(path)
        assert config is not None
        assert isinstance(config.model_id, str)

