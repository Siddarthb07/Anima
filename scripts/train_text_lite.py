"""
Train GoEmotions text probe on the tiny default HF model (low RAM).
Use when distilgpt2 / 7B models fail with paging-file errors on Windows.

  python scripts/train_text_lite.py --max-samples 60
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ANIMA_FORCE_CPU", "1")

from core.defaults import DEFAULT_CAUSAL_LM
from probes.train_text import train_text_probe
from probes.zoo_io import probe_slug, save_probe_bundle


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--max-samples", type=int, default=60)
    p.add_argument("--epochs", type=int, default=8)
    args = p.parse_args()
    model = DEFAULT_CAUSAL_LM
    print(f"Training text probe on {model} (max_samples={args.max_samples})")
    probe, meta = train_text_probe(
        model,
        max_samples=args.max_samples,
        epochs=args.epochs,
        device="cpu",
    )
    slug = probe_slug(model)
    path = save_probe_bundle(probe, slug, meta, suffix="_text")
    print("Saved", path)


if __name__ == "__main__":
    main()
