"""
Unit tests for inference CLI argument parsing, dtype conversions, and generation parameter plumbing.
"""

import argparse
from unittest.mock import MagicMock

import pytest
import torch

from inference import (
    build_parser,
    generate_response,
    parse_args,
    parse_dtype,
)


def test_parse_dtype() -> None:
    assert parse_dtype("float16") == torch.float16
    assert parse_dtype("bfloat16") == torch.bfloat16
    assert parse_dtype("float32") == torch.float32

    with pytest.raises(argparse.ArgumentTypeError):
        parse_dtype("int8")

    with pytest.raises(argparse.ArgumentTypeError):
        parse_dtype("unsupported_type")


def test_default_cli_args() -> None:
    parser = build_parser()
    args = parser.parse_args([])

    assert args.model == "google/gemma-3-270m-it"
    assert args.max_len == 2048
    assert args.dtype == torch.bfloat16
    assert args.do_sample is True
    assert args.temperature == 0.7
    assert args.top_p == 0.95
    assert args.top_k == 65
    assert args.repetition_penalty == 1.5


def test_custom_cli_args() -> None:
    cli_input = [
        "--model",
        "custom/test-gemma",
        "--max-len",
        "1024",
        "--dtype",
        "float16",
        "--no-sample",
        "--temperature",
        "0.2",
        "--top-p",
        "0.8",
        "--top-k",
        "40",
        "--repetition-penalty",
        "1.1",
    ]
    args = parse_args(cli_input)

    assert args.model == "custom/test-gemma"
    assert args.max_len == 1024
    assert args.dtype == torch.float16
    assert args.do_sample is False
    assert args.temperature == 0.2
    assert args.top_p == 0.8
    assert args.top_k == 40
    assert args.repetition_penalty == 1.1


def test_custom_cli_args_underscore_and_flag_variations() -> None:
    cli_input = [
        "--do_sample",
        "false",
        "--top_p",
        "0.9",
        "--top_k",
        "50",
        "--repetition_penalty",
        "1.3",
        "--max_len",
        "512",
    ]
    args = parse_args(cli_input)

    assert args.do_sample is False
    assert args.top_p == 0.9
    assert args.top_k == 50
    assert args.repetition_penalty == 1.3
    assert args.max_len == 512


def test_generate_response_calls_model_with_correct_parameters() -> None:
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.return_value = MagicMock(
        input_ids=torch.tensor([[1, 2]]),
        to=lambda dev: {"input_ids": torch.tensor([[1, 2]])},
    )

    generate_response(
        model=mock_model,
        tokenizer=mock_tokenizer,
        prompt="Bonjour le monde",
        max_len=2048,
        dtype=torch.float32,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=65,
        repetition_penalty=1.5,
        streamer=None,
    )

    assert mock_model.generate.called
    _, kwargs = mock_model.generate.call_args
    assert kwargs.get("do_sample") is True
    assert kwargs.get("temperature") == 0.7
    assert kwargs.get("top_p") == 0.95
    assert kwargs.get("top_k") == 65
    assert kwargs.get("repetition_penalty") == 1.5


def test_generate_response_with_chat_template() -> None:
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}"
    mock_tokenizer.apply_chat_template.return_value = MagicMock(
        input_ids=torch.tensor([[1, 2]]),
        to=lambda dev: MagicMock(input_ids=torch.tensor([[1, 2]])),
    )

    generate_response(
        model=mock_model,
        tokenizer=mock_tokenizer,
        prompt="Bonjour avec template",
        max_len=1024,
        dtype=torch.bfloat16,
        do_sample=True,
    )

    assert mock_tokenizer.apply_chat_template.called
    assert mock_model.generate.called


def test_generate_response_cpu_float16_safety() -> None:
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.return_value = MagicMock(
        input_ids=torch.tensor([[1, 2]]),
        to=lambda dev: {"input_ids": torch.tensor([[1, 2]])},
    )

    # Should not raise RuntimeError on CPU float16
    generate_response(
        model=mock_model,
        tokenizer=mock_tokenizer,
        prompt="Bonjour test float16",
        max_len=512,
        dtype=torch.float16,
    )
    assert mock_model.generate.called


def test_find_tokenizer_path(tmp_path) -> None:
    from inference import find_tokenizer_path

    # Case 1: Tokenizer directly in directory
    dir1 = tmp_path / "model1"
    dir1.mkdir()
    (dir1 / "tokenizer.json").write_text("{}")
    assert find_tokenizer_path(str(dir1)) == str(dir1)

    # Case 2: Tokenizer in tokenizer_checkpoint subdirectory
    dir2 = tmp_path / "model2"
    dir2.mkdir()
    sub = dir2 / "tokenizer_checkpoint"
    sub.mkdir()
    (sub / "tokenizer.json").write_text("{}")
    assert find_tokenizer_path(str(dir2)) == str(sub)

    # Case 3: Tokenizer in sibling directory
    parent = tmp_path / "pretrain_run"
    parent.mkdir()
    ckpt = parent / "checkpoint-step-100"
    ckpt.mkdir()
    sib = parent / "tokenizer_checkpoint"
    sib.mkdir()
    (sib / "tokenizer.json").write_text("{}")
    assert find_tokenizer_path(str(ckpt)) == str(sib)


def test_load_model_and_tokenizer_associates_chat_template(tmp_path, monkeypatch) -> None:
    import json
    from unittest.mock import MagicMock

    from inference import load_model_and_tokenizer
    from src.dataset import GEMMA_CHAT_TEMPLATE

    ckpt_dir = tmp_path / "checkpoint-no-template"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text(json.dumps({"architectures": ["Gemma3ForCausalLM"]}))
    (ckpt_dir / "tokenizer.json").write_text("{}")

    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.eos_token_id = 2

    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.generation_config = MagicMock()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: mock_tokenizer)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: mock_model)

    model, tokenizer, streamer = load_model_and_tokenizer(str(ckpt_dir), dtype=torch.float32, max_len=512)

    assert tokenizer.chat_template == GEMMA_CHAT_TEMPLATE
    assert model == mock_model
    assert streamer is not None
