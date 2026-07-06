"""
Train GoEmotions text probes for all configured models (or a subset).

CPU default: distilgpt2 + smaller family proxies (Llama-3.2-1B, Qwen2.5-0.5B, Gemma-2-2b).
7B/8B/9B rows need GPU or ANIMA_TRAIN_LARGE=1 + ANIMA_LOAD_8BIT=1.

Usage:
  python scripts/train_text_zoo_all.py
  python scripts/train_text_zoo_all.py --models distilgpt2 meta-llama/Llama-3.2-1B-Instruct
  set ANIMA_TRAIN_LARGE=1 & set ANIMA_LOAD_8BIT=1 & python scripts/train_text_zoo_all.py --tier large
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

# Stable CPU / low-memory loading before torch import side effects
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ANIMA_FORCE_CPU", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.layer_config import LAYER_CONFIG
from probes.train_text import train_text_probe
from probes.zoo_io import probe_slug, save_probe_bundle

CPU_MODELS = [
    "distilgpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]

LARGE_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2-7B-Instruct",
    "google/gemma-2-9b-it",
]


def _can_train_large() -> bool:
    if os.environ.get("ANIMA_TRAIN_LARGE") != "1":
        return False
    import torch

    return torch.cuda.is_available() or os.environ.get("ANIMA_LOAD_8BIT") == "1"


def train_one(
    model: str,
    *,
    max_samples: int,
    epochs: int,
    device: str,
) -> dict:
    print(f"\n=== train-text: {model} ===", flush=True)
    probe, meta = train_text_probe(
        model,
        max_samples=max_samples,
        epochs=epochs,
        device=device,
    )
    slug = probe_slug(model)
    path = save_probe_bundle(probe, slug, meta, suffix="_text")
    print(f"Saved {path}", flush=True)
    return {"model": model, "path": str(path), **meta}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=["cpu", "large", "all"], default="cpu")
    p.add_argument("--models", nargs="*", help="HF ids (override tier list)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.models:
        models = list(args.models)
    elif args.tier == "large":
        models = LARGE_MODELS if _can_train_large() else []
        if not models:
            print("Skip large tier: set ANIMA_TRAIN_LARGE=1 and use a CUDA GPU (or 8-bit + GPU).")
            return
    elif args.tier == "all":
        models = CPU_MODELS + (LARGE_MODELS if _can_train_large() else [])
    else:
        models = CPU_MODELS

    results = []
    for model in models:
        if model not in LAYER_CONFIG:
            print(f"Skip unknown model: {model}")
            continue
        cfg = LAYER_CONFIG[model]
        if cfg.get("requires_gpu") and not _can_train_large():
            print(f"Skip {model} (requires_gpu; set ANIMA_TRAIN_LARGE=1 on a GPU machine)")
            results.append({"model": model, "status": "skipped", "reason": "requires_gpu"})
            continue
        if cfg.get("gated") and not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            print(f"Skip {model} (gated — run huggingface-cli login or set HF_TOKEN)")
            results.append({"model": model, "status": "skipped", "reason": "gated_hf"})
            continue
        max_s = args.max_samples
        if max_s is None:
            # Scale by model size; Qwen 0.5B / distilgpt2 tolerate 1.5–2k on 16 GB CPU.
            if cfg["hidden_dim"] <= 768:
                max_s = 1500
            elif cfg["hidden_dim"] <= 1024:
                max_s = 2000
            else:
                max_s = 1000
        ep = args.epochs if args.epochs is not None else (15 if cfg["hidden_dim"] <= 1024 else 12)
        try:
            if cfg.get("requires_gpu"):
                os.environ["ANIMA_FORCE_CPU"] = "0"
                device = "auto"
            else:
                os.environ["ANIMA_FORCE_CPU"] = "1"
                device = args.device
            meta = train_one(model, max_samples=max_s, epochs=ep, device=device)
            meta["status"] = "ok"
            results.append(meta)
        except Exception as exc:
            print(f"FAILED {model}: {exc}", flush=True)
            results.append({"model": model, "status": "error", "message": str(exc)})
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    report = ROOT / "probes" / "zoo" / "train_text_report.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
