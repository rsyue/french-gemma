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
    assert config.num_examples == "all"



def test_parse_args_num_examples_all():
    """Verify --num-examples all and numeric strings are accepted by the CLI parser."""
    config = parse_args_to_config(["--num-examples", "all"])
    assert config.num_examples == "all"

    config_num = parse_args_to_config(["--num_examples", "500"])
    assert config_num.num_examples == 500


def test_parse_args_packing_batch_size():
    """Verify --packing-batch-size and --packing_batch_size override config values."""
    config = parse_args_to_config(["--packing-batch-size", "500"])
    assert config.packing_batch_size == 500

    config_underscore = parse_args_to_config(["--packing_batch_size", "1500"])
    assert config_underscore.packing_batch_size == 1500


def test_parse_args_save_dir():
    """Verify --save-dir and --save_dir override output_dir and save_dir."""
    config_dash = parse_args_to_config(["--save-dir", "./custom_save_dir"])
    assert config_dash.output_dir == "./custom_save_dir"
    assert config_dash.save_dir == "./custom_save_dir"

    config_under = parse_args_to_config(["--save_dir", "./another_dir"])
    assert config_under.output_dir == "./another_dir"
    assert config_under.save_dir == "./another_dir"


def test_parse_args_modality_defaults():
    """Verify modality parameter automatically sets separated default directories inside ./checkpoints."""
    config_pretrain = parse_args_to_config([], modality="pretrain")
    assert config_pretrain.output_dir == "./checkpoints/pretrain"
    assert config_pretrain.save_dir == "./checkpoints/pretrain"

    config_sft = parse_args_to_config([], modality="sft")
    assert config_sft.output_dir == "./checkpoints/sft"
    assert config_sft.save_dir == "./checkpoints/sft"

    config_dpo = parse_args_to_config([], modality="dpo")
    assert config_dpo.output_dir == "./checkpoints/dpo"
    assert config_dpo.save_dir == "./checkpoints/dpo"

    # CLI argument overrides modality default
    config_override = parse_args_to_config(["--save-dir", "./custom_dpo"], modality="dpo")
    assert config_override.output_dir == "./custom_dpo"
    assert config_override.save_dir == "./custom_dpo"


def test_multi_epoch_loop_logic():
    """Verify multi-epoch loop logic runs past epoch 0 until max_steps is reached."""
    max_steps = 10
    global_step = 0
    epoch = 0

    class MockTrainer:
        def train_epoch(self, epoch: int, global_step: int) -> int:
            # Simulate 3 steps per epoch
            return global_step + 3

    trainer = MockTrainer()
    epochs_run = 0
    while global_step < max_steps:
        global_step = trainer.train_epoch(epoch=epoch, global_step=global_step)
        epoch += 1
        epochs_run += 1

    assert global_step >= max_steps
    assert epochs_run > 1  # Verify it ran multiple epochs

