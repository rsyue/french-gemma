"""
Inference and Chat Interface for French Gemma 3.

This module provides an interactive CLI chat loop utilizing text streaming decoding
and configurable generation sampling parameters.
"""

import argparse
import sys
from typing import Any, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse string representations into corresponding PyTorch dtypes."""
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float32":
        return torch.float32
    raise argparse.ArgumentTypeError(f"Unsupported dtype: {dtype_str}")


def str_to_bool(val: Any) -> bool:
    """Convert string or boolean input to boolean."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        val_lower = val.lower()
        if val_lower in ("yes", "true", "t", "y", "1"):
            return True
        elif val_lower in ("no", "false", "f", "n", "0"):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{val}'.")


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the CLI argument parser for inference."""
    parser = argparse.ArgumentParser(description="French Gemma 3 Inference & Interactive Chat")
    parser.add_argument(
        "--model",
        help="The model identifier or local directory path",
        default="google/gemma-3-270m-it",
    )
    parser.add_argument(
        "--max-len",
        "--max_len",
        dest="max_len",
        help="The context length for generation",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--dtype",
        help="The dtype to use for model and tokens (float16, bfloat16, float32)",
        type=parse_dtype,
        default=torch.bfloat16,
    )
    parser.add_argument(
        "--do-sample",
        "--do_sample",
        dest="do_sample",
        help="Whether to use sampling instead of greedy decoding (default: True)",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--no-sample",
        "--no_sample",
        dest="do_sample",
        help="Disable sampling and use greedy decoding",
        action="store_false",
    )
    parser.add_argument(
        "--temperature",
        help="The value used to modulate next token probabilities (default: 0.7)",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--top-p",
        "--top_p",
        dest="top_p",
        help="Nucleus sampling probability mass threshold (default: 0.95)",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--top-k",
        "--top_k",
        dest="top_k",
        help="Top-k tokens to consider for sampling (default: 65)",
        type=int,
        default=65,
    )
    parser.add_argument(
        "--repetition-penalty",
        "--repetition_penalty",
        dest="repetition_penalty",
        help="Penalty for repeating tokens during generation (default: 1.5)",
        type=float,
        default=1.5,
    )
    return parser


def parse_args(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments into a namespace."""
    parser = build_parser()
    return parser.parse_args(args_list)


def load_model_and_tokenizer(
    model_id: str,
    dtype: torch.dtype,
    max_len: int,
) -> Tuple[Any, Any, TextStreamer]:
    """Load model, tokenizer, and configure streaming decoder."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, is_fast=True, truncation=True, max_length=max_len
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", dtype=dtype
    ).eval()  # type: ignore[no-untyped-call]

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    return model, tokenizer, streamer


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_len: int,
    dtype: torch.dtype,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 65,
    repetition_penalty: float = 1.5,
    streamer: Optional[TextStreamer] = None,
) -> Any:
    """Generate model response for a prompt using specified sampling parameters."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    try:
        has_template = getattr(tokenizer, "chat_template", None) is not None
    except Exception:
        has_template = False

    device = getattr(model, "device", "cpu")

    if has_template:
        inputs_any: Any = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs_any.to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

    input_length = inputs.input_ids.shape[1] if hasattr(inputs, "input_ids") else inputs["input_ids"].shape[1]
    max_new_tokens = max(1, max_len - input_length)

    device_type = "cuda" if "cuda" in str(device) else ("mps" if "mps" in str(device) else "cpu")
    autocast_enabled = (device_type == "cuda") or (device_type == "cpu" and dtype == torch.bfloat16)
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
    return output


def chat_loop(
    model: Any,
    tokenizer: Any,
    streamer: TextStreamer,
    max_len: int,
    dtype: torch.dtype,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> None:
    """Run interactive CLI chat loop."""
    print(f"Model ready on device: {model.device}. Enter prompts (Ctrl+C to exit):")
    while True:
        try:
            prompt = input("Prompt: ")
            if not prompt.strip():
                continue
            print()
            generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_len=max_len,
                dtype=dtype,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                streamer=streamer,
            )
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session.")
            break


def main(args_list: Optional[List[str]] = None) -> None:
    """Main execution function for inference script."""
    args = parse_args(args_list)
    model, tokenizer, streamer = load_model_and_tokenizer(
        model_id=args.model,
        dtype=args.dtype,
        max_len=args.max_len,
    )
    chat_loop(
        model=model,
        tokenizer=tokenizer,
        streamer=streamer,
        max_len=args.max_len,
        dtype=args.dtype,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
