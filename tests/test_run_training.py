"""
Unit Tests for training script configuration loading and CLI overrides.
"""

import os
import sys
import tempfile
import yaml

# Add scripts directory to sys.path to import run_training
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_training


def test_parse_and_load_config_defaults():
    # When no config file is passed, or when the default mlx_config.yaml doesn't exist,
    # the function should return a TrainingConfig filled with defaults.
    config = run_training.parse_and_load_config(["--config", "non_existent_config.yaml"])
    assert config.model_id == "google/gemma-3-270m-it"
    assert config.learning_rate == 1.0e-4
    assert config.device == "cpu"


def test_parse_and_load_config_yaml():
    # When a YAML config is provided but lacks some fields, it should fall back to reasonable defaults.
    yaml_content = {
        "model_id": "custom-gemma-model",
        "batch_size": 4,
        "freeze_schedule": {0: [1, 2]},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        # Load and check values
        config = run_training.parse_and_load_config(["--config", config_path])
        assert config.model_id == "custom-gemma-model"
        assert config.batch_size == 4
        assert config.freeze_schedule == {0: [1, 2]}
        # Missing fields should fall back to dataclass defaults
        assert config.learning_rate == 1.0e-4
        assert config.device == "cpu"
        assert config.max_eval_batches == 20


def test_parse_and_load_config_cli_overrides():
    # CLI arguments should override the YAML settings
    yaml_content = {
        "model_id": "custom-gemma-model",
        "learning_rate": 1.0e-4,
        "compile": False,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        # Override model_id, learning_rate, and compile via CLI
        config = run_training.parse_and_load_config([
            "--config", config_path,
            "--model-id", "cli-gemma-model",
            "--learning-rate", "2.0e-5",
            "--compile", "true",
        ])
        assert config.model_id == "cli-gemma-model"
        assert config.learning_rate == 2.0e-5
        assert config.compile is True
        # Fields not specified in CLI or YAML should still have dataclass defaults
        assert config.batch_size == 2
