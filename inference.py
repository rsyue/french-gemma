"""
Inference and Chat Interface for French Gemma 3.

This script loads a pretrained Gemma 3 model and presents an interactive CLI chat loop
utilizing text streaming decoding.
"""

import argparse
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def parse_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float32":
        return torch.float32
    raise argparse.ArgumentTypeError(f"Unsupported dtype: {dtype_str}")

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="The model to use", default="google/gemma-3-270m-it")
parser.add_argument("--max-len", help="The context length for generation", type=int, default=2048)
parser.add_argument("--dtype", help="The dtype to use for model and tokens", type=parse_dtype, default=torch.bfloat16)
args = parser.parse_args()

model_id = args.model
max_len = args.max_len
dtype = args.dtype

tokenizer = AutoTokenizer.from_pretrained(model_id, is_fast=True, truncation=True, max_length=max_len)

model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", dtype=dtype
).eval()  # type: ignore[no-untyped-call]
model.generation_config.pad_token_id = tokenizer.eos_token_id
print(f"Using device: {model.device}")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


def main() -> None:
    while True:
        prompt = input("Prompt: ")
        print()

        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]

        # Check if chat template is available, fallback to direct tokenization if not
        try:
            has_template = tokenizer.chat_template is not None
        except Exception:
            has_template = False

        if has_template:
            inputs_any: Any = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs_any.to(model.device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.autocast(device_type=str(model.device), dtype=dtype):
            with torch.inference_mode():
                _ = model.generate(**inputs, streamer=streamer, max_new_tokens=(max_len - inputs.input_ids.shape[1]))
        print()


if __name__ == "__main__":
    main()
