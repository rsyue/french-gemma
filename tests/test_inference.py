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


def test_generate_response_without_chat_template_warns_and_runs(caplog) -> None:
    import logging

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.return_value = MagicMock(
        input_ids=torch.tensor([[1, 2]]),
        to=lambda dev: {"input_ids": torch.tensor([[1, 2]])},
    )

    with caplog.at_level(logging.WARNING):
        output = generate_response(
            model=mock_model,
            tokenizer=mock_tokenizer,
            prompt="Prompt without template",
            max_len=128,
            dtype=torch.float32,
        )

    assert output is not None
    assert mock_model.generate.called
    assert any("No chat template" in record.message for record in caplog.records)


def test_inference_loads_checkpoint_directory_with_saved_tokenizer(tmp_path) -> None:
    import os

    from inference import load_model_and_tokenizer
    from src.dataset import GEMMA_CHAT_TEMPLATE, train_custom_tokenizer
    from src.model import FrenchGemmaModel

    ckpt_dir = str(tmp_path / "sft_checkpoint_dir")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Save dummy model config and weights
    model = FrenchGemmaModel(model_id="google/gemma-3-270m-it", vocab_size=200)
    model.config.architectures = ["Gemma3ForCausalLM"]
    model.config.save_pretrained(ckpt_dir)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))

    # 2. Save tokenizer with chat template
    tok = train_custom_tokenizer(["Sample text for tokenizer"], vocab_size=200, save_dir=ckpt_dir)
    assert tok.chat_template == GEMMA_CHAT_TEMPLATE

    # 3. Load via inference load_model_and_tokenizer
    loaded_model, loaded_tok, streamer = load_model_and_tokenizer(
        model_id=ckpt_dir,
        dtype=torch.float32,
        max_len=512,
    )
    assert loaded_model is not None
    assert loaded_tok is not None
    assert loaded_tok.chat_template is not None
    assert "<start_of_turn>" in loaded_tok.chat_template

