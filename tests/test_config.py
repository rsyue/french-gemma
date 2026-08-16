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
    assert config.num_examples == "all"
    assert config.warmup_ratio == 0.03
    assert config.T_mult == 2
    assert config.num_cycles == 1
    assert config.eta_min_ratio == 0.01



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
    expected_packing_sizes = {
        "amd_config.yaml": 10000,
        "mlx_config.yaml": 2000,
        "nvidia_config.yaml": 1000,
    }
    for filename in yaml_files:
        path = os.path.join(config_dir, filename)
        config = TrainingConfig.from_yaml(path)
        assert config is not None
        assert isinstance(config.model_id, str)
        assert config.num_examples is not None
        if filename in expected_packing_sizes:
            assert config.packing_batch_size == expected_packing_sizes[filename]


def test_specific_repo_configs():
    config_dir = "configs"
    mlx_cfg = TrainingConfig.from_yaml(os.path.join(config_dir, "mlx_config.yaml"))
    assert mlx_cfg.vocab_size == 25000

    nvidia_cfg = TrainingConfig.from_yaml(os.path.join(config_dir, "nvidia_config.yaml"))
    assert nvidia_cfg.vocab_size == 25000

    amd_cfg = TrainingConfig.from_yaml(os.path.join(config_dir, "amd_config.yaml"))
    assert amd_cfg.vocab_size == 50000
    assert amd_cfg.max_sequence_length == 4096
    assert amd_cfg.batch_size == 8
    assert amd_cfg.gradient_accumulation_steps == 16
    assert amd_cfg.num_examples == 1000
    assert amd_cfg.max_steps == 25
    assert amd_cfg.warmup_steps == 1
    assert amd_cfg.seed == 42
    assert amd_cfg.save_dir == "./checkpoints/pretrain"

    sft_cfg = TrainingConfig.from_yaml(os.path.join(config_dir, "sft_config.yaml"))
    assert sft_cfg.max_sequence_length == 4096
    assert sft_cfg.save_dir == "./checkpoints/sft"
    assert sft_cfg.output_dir == "./checkpoints/sft"

    dpo_cfg = TrainingConfig.from_yaml(os.path.join(config_dir, "dpo_config.yaml"))
    assert dpo_cfg.max_sequence_length == 4096
    assert dpo_cfg.save_dir == "./checkpoints/dpo"
    assert dpo_cfg.output_dir == "./checkpoints/dpo"
    assert dpo_cfg.dpo_beta == 0.1


def test_config_save_dir_synchronization():
    cfg1 = TrainingConfig(save_dir="./custom_save")
    assert cfg1.output_dir == "./custom_save"
    assert cfg1.save_dir == "./custom_save"

    cfg2 = TrainingConfig(output_dir="./custom_out")
    assert cfg2.output_dir == "./custom_out"
    assert cfg2.save_dir == "./custom_out"




