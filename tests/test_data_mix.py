"""
Unit and integration tests for YAML dataset mix configurations, percentage weighting,
and multi-dataset loading.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from src.config import DatasetMixEntry, TrainingConfig
from src.dataset import load_dataset_mix, prepare_and_pack_data, train_custom_tokenizer
from train.cli import parse_args_to_config


def test_dataset_mix_entry_creation() -> None:
    entry = DatasetMixEntry(
        dataset_path="wikimedia/wikipedia",
        dataset_name="20231101.fr",
        percentage=50.0,
        split="train",
    )
    assert entry.dataset_path == "wikimedia/wikipedia"
    assert entry.dataset_name == "20231101.fr"
    assert entry.percentage == 50.0
    assert entry.split == "train"


def test_training_config_parses_inline_data_mix() -> None:
    yaml_content = """
model_id: "google/gemma-3-270m-it"
num_examples: 1000
data_mix:
  - dataset_path: "wikimedia/wikipedia"
    dataset_name: "20231101.fr"
    percentage: 50.0
  - dataset_path: "oscar-corpus/OSCAR-2201"
    dataset_name: "fr"
    percentage: 30.0
  - dataset_path: "c4"
    dataset_name: "fr"
    percentage: 20.0
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config_path = f.name

    try:
        config = TrainingConfig.from_yaml(config_path)
        assert config.data_mix is not None
        assert len(config.data_mix) == 3
        assert config.data_mix[0].dataset_path == "wikimedia/wikipedia"
        assert config.data_mix[0].percentage == 50.0
        assert config.data_mix[1].dataset_path == "oscar-corpus/OSCAR-2201"
        assert config.data_mix[1].percentage == 30.0
        assert config.data_mix[2].dataset_path == "c4"
        assert config.data_mix[2].percentage == 20.0
    finally:
        os.remove(config_path)


def test_training_config_parses_standalone_data_mix_file() -> None:
    mix_yaml_content = """
data_mix:
  - path: "corpus_a"
    name: "subset1"
    percentage: 60
  - path: "corpus_b"
    percentage: 40
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f_mix:
        f_mix.write(mix_yaml_content)
        f_mix.flush()
        mix_path = f_mix.name

    main_yaml_content = f"""
model_id: "google/gemma-3-270m-it"
data_mix: "{mix_path}"
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f_main:
        f_main.write(main_yaml_content)
        f_main.flush()
        main_path = f_main.name

    try:
        config = TrainingConfig.from_yaml(main_path)
        assert config.data_mix is not None
        assert len(config.data_mix) == 2
        assert config.data_mix[0].dataset_path == "corpus_a"
        assert config.data_mix[0].dataset_name == "subset1"
        assert config.data_mix[0].percentage == 60.0
        assert config.data_mix[1].dataset_path == "corpus_b"
        assert config.data_mix[1].percentage == 40.0
    finally:
        os.remove(mix_path)
        os.remove(main_path)


def test_load_dataset_mix_proportions_and_deterministic_seeding() -> None:
    mix = [
        DatasetMixEntry(dataset_path="wiki", dataset_name="fr", percentage=50.0),
        DatasetMixEntry(dataset_path="news", dataset_name="fr", percentage=30.0),
        DatasetMixEntry(dataset_path="books", dataset_name="fr", percentage=20.0),
    ]

    mock_datasets = {
        "wiki/fr": [f"wiki_doc_{i}" for i in range(100)],
        "news/fr": [f"news_doc_{i}" for i in range(100)],
        "books/fr": [f"books_doc_{i}" for i in range(100)],
    }

    def mock_load_french_dataset(dataset_path, dataset_name=None, split="train", fallback_texts=None):
        key = f"{dataset_path}/{dataset_name}" if dataset_name else dataset_path
        data = mock_datasets.get(key, [])
        if "[" in split and ":" in split:
            slice_part = split.split("[")[1].split("]")[0]
            if slice_part.startswith(":"):
                count = int(slice_part[1:])
                return data[:count]
        return data

    with patch("src.dataset.load_french_dataset", side_effect=mock_load_french_dataset):
        # Request 100 total examples
        texts1 = load_dataset_mix(data_mix=mix, total_examples=100, seed=42)
        texts2 = load_dataset_mix(data_mix=mix, total_examples=100, seed=42)

        # Deterministic seeding check
        assert texts1 == texts2
        assert len(texts1) == 100

        # Check proportion counts
        wiki_count = sum(1 for t in texts1 if t.startswith("wiki_doc_"))
        news_count = sum(1 for t in texts1 if t.startswith("news_doc_"))
        books_count = sum(1 for t in texts1 if t.startswith("books_doc_"))

        assert wiki_count == 50
        assert news_count == 30
        assert books_count == 20


def test_load_dataset_mix_validation_and_edge_cases() -> None:
    # Empty data mix
    with pytest.raises(ValueError, match="data_mix cannot be empty"):
        load_dataset_mix(data_mix=[])

    # Negative percentage
    with pytest.raises(ValueError, match="percentage must be non-negative"):
        load_dataset_mix(
            data_mix=[DatasetMixEntry(dataset_path="wiki", percentage=-10.0)]
        )

    # All zeros
    with pytest.raises(ValueError, match="Total percentage sum must be greater than zero"):
        load_dataset_mix(
            data_mix=[
                DatasetMixEntry(dataset_path="wiki", percentage=0.0),
                DatasetMixEntry(dataset_path="news", percentage=0.0),
            ]
        )


def test_cli_data_mix_argument_parsing() -> None:
    mix_yaml = """
data_mix:
  - path: "cli_wiki"
    percentage: 70
  - path: "cli_news"
    percentage: 30
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(mix_yaml)
        f.flush()
        mix_path = f.name

    try:
        config = parse_args_to_config(["--data-mix", mix_path])
        assert config.data_mix is not None
        assert len(config.data_mix) == 2
        assert config.data_mix[0].dataset_path == "cli_wiki"
        assert config.data_mix[0].percentage == 70.0
        assert config.data_mix[1].dataset_path == "cli_news"
        assert config.data_mix[1].percentage == 30.0
    finally:
        os.remove(mix_path)


def test_dataset_mix_end_to_end_packing() -> None:
    mix = [
        DatasetMixEntry(dataset_path="corpus1", percentage=50.0),
        DatasetMixEntry(dataset_path="corpus2", percentage=50.0),
    ]

    mock_texts = {
        "corpus1": ["Bonjour de corpus un numéro 1.", "Deuxième texte de corpus un."],
        "corpus2": ["Bonjour de corpus deux numéro 1.", "Deuxième texte de corpus deux."],
    }

    def mock_loader(dataset_path, dataset_name=None, split="train", fallback_texts=None):
        return mock_texts.get(dataset_path, ["Fallback text."])

    with tempfile.TemporaryDirectory() as tmpdir:
        tok_dir = os.path.join(tmpdir, "tok")
        cache_bin = os.path.join(tmpdir, "cache.bin")

        with patch("src.dataset.load_french_dataset", side_effect=mock_loader):
            texts = load_dataset_mix(mix, total_examples=4, seed=42)
            assert len(texts) == 4

            tokenizer = train_custom_tokenizer(texts, vocab_size=100, save_dir=tok_dir)
            prepare_and_pack_data(texts, tokenizer, cache_bin, packing_batch_size=2)

            assert os.path.exists(cache_bin)
            assert os.path.exists(cache_bin + ".ready")
            assert os.path.getsize(cache_bin) > 0
