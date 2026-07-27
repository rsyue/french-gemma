"""
Unit Tests for training script configuration loading and CLI overrides.
"""

import os
import sys
import tempfile

import yaml

# Add scripts directory to sys.path to import training_example
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
import training_example  # type: ignore[import-not-found]


def test_parse_and_load_config_defaults():
    config = training_example.parse_and_load_config(["--config", "non_existent_config.yaml"])
    assert config.model_id == "google/gemma-3-270m-it"
    assert config.learning_rate == 1.0e-4
    assert config.device == "cpu"


def test_parse_and_load_config_yaml():
    yaml_content = {
        "model_id": "custom-gemma-model",
        "batch_size": 4,
        "freeze_schedule": {0: [1, 2]},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        config = training_example.parse_and_load_config(["--config", config_path])
        assert config.model_id == "custom-gemma-model"
        assert config.batch_size == 4
        assert config.freeze_schedule == {0: [1, 2]}
        assert config.learning_rate == 1.0e-4
        assert config.device == "cpu"
        assert config.max_eval_batches == 20


def test_parse_and_load_config_cli_overrides():
    yaml_content = {
        "model_id": "custom-gemma-model",
        "learning_rate": 1.0e-4,
        "compile": False,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        config = training_example.parse_and_load_config([
            "--config", config_path,
            "--model-id", "cli-gemma-model",
            "--learning-rate", "2.0e-5",
            "--compile", "true",
        ])
        assert config.model_id == "cli-gemma-model"
        assert config.learning_rate == 2.0e-5
        assert config.compile is True
        assert config.batch_size == 2


def test_prepare_and_pack_data():
    import numpy as np

    from src.dataset import train_custom_tokenizer

    mock_texts = [
        "Texte un de test.",
        "Deuxième texte de test.",
        "Troisième exemple en français.",
        "Un dernier texte."
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        tok_dir = os.path.join(tmpdir, "tok")
        tokenizer = train_custom_tokenizer(mock_texts, vocab_size=100, save_dir=tok_dir)
        cache_path = os.path.join(tmpdir, "packed.bin")

        training_example.prepare_and_pack_data(
            texts=mock_texts,
            tokenizer=tokenizer,
            cache_path=cache_path,
            packing_batch_size=2,
            packing_log_interval=1,
        )

        assert os.path.exists(cache_path)
        assert os.path.getsize(cache_path) > 0
        data = np.fromfile(cache_path, dtype=np.uint32)
        assert len(data) > 0
