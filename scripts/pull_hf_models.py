"""Pre-download Hugging Face model weights into the hub cache (for Docker volumes)."""

from __future__ import annotations

import argparse
import os

# CPU-friendly models with shipped or trainable probes
DEFAULT_MODELS = [
    "distilgpt2",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "hf-internal-testing/tiny-random-gpt2",
]


def pull(model_id: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Pulling {model_id}...")
    AutoTokenizer.from_pretrained(model_id, use_fast=False)
    AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    print(f"  ok: {model_id}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = p.parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    for m in args.models:
        try:
            pull(m)
        except Exception as exc:
            print(f"  FAIL {m}: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
