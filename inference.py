# Inference script for Gemma 3
# Chat with the model

# @author: Richard Yue

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="The model to use", default="google/gemma-3-270m-it")
parser.add_argument("--max-len", help="The context length for generation", default=2048)
parser.add_argument("--dtype", help="The dtype to use for model and tokens", default=torch.bfloat16)
args = parser.parse_args()

model_id = args.model
max_len = args.max_len
dtype = args.dtype

tokenizer = AutoTokenizer.from_pretrained(model_id, is_fast=True, truncation=True, max_length=max_len)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=dtype).eval()
model.generation_config.pad_token_id = tokenizer.eos_token_id
print(f"Using device: {model.device}")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

def main():
    while True:
        prompt = input("Prompt: ")
        print()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.autocast(device_type=str(model.device), dtype=dtype):
            with torch.inference_mode():
                _ = model.generate(**inputs, streamer=streamer, max_new_tokens=(max_len - inputs.input_ids.dim()))
        print()

if __name__ == "__main__":
    main()
