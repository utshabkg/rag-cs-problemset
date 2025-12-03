"""
Script to download and load selected Huggingface models for evaluation.
Models:
- Llama-3-8B (meta-llama/Meta-Llama-3-8B)
- Mistral-7B (mistralai/Mistral-7B-v0.1)
- Qwen-7B (Qwen/Qwen-7B)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

MODELS = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "qwen-7b": "Qwen/Qwen-7B"
}

MODEL_DIR = '/media/12TB/shared/models'


def download_and_load(model_name, device="cuda"):
    model_id = MODELS[model_name]
    print(f"Downloading/loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=MODEL_DIR, torch_dtype=torch.float16, device_map="auto")
    model.to(device)
    return model, tokenizer

if __name__ == "__main__":
    for name in MODELS:
        model, tokenizer = download_and_load(name)
        print(f"Loaded {name} model.")
