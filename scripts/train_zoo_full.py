"""
Train text + brain probes for every model in core/layer_config.py (OSS zoo build).

  python scripts/download_narratives_minimal.py
  python scripts/train_zoo_full.py --tier cpu
  ANIMA_TRAIN_LARGE=1 python scripts/train_zoo_full.py --tier all   # GPU machine

Ollama is not supported — use matching Hugging Face ids (see scripts/ollama_to_hf.json).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.layer_config import LAYER_CONFIG
from core.extractor import ActivationExtractor
from probes.train import train_probe
from probes.train_text import train_text_probe
from probes.zoo_io import probe_slug, save_probe_bundle

SKIP_IDS = {"hf-internal-testing/tiny-random-gpt2"}


def _models_for_tier(tier: str) -> list[str]:
    all_ids = [m for m in LAYER_CONFIG if m not in SKIP_IDS]
    cpu = [m for m in all_ids if not LAYER_CONFIG[m].get("requires_gpu")]
    large = [m for m in all_ids if LAYER_CONFIG[m].get("requires_gpu")]
    if tier == "cpu":
        return cpu
    if tier == "large":
        return large
    return cpu + large


def _narratives_root() -> str:
    env = os.environ.get("NARRATIVES_ROOT", "")
    if env and Path(env).is_dir():
        return env
    fallback = ROOT / "data" / "narratives_minimal"
    if fallback.is_dir():
        return str(fallback)
    raise FileNotFoundError("Run: python scripts/download_narratives_minimal.py")


def _hf_authenticated() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def _can_large() -> bool:
    if os.environ.get("ANIMA_TRAIN_LARGE") != "1":
        return False
    import torch

    return torch.cuda.is_available()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=["cpu", "large", "all"], default="cpu")
    p.add_argument("--models", nargs="*", help="HF ids (override tier list)")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--epochs-text", type=int, default=None)
    p.add_argument("--epochs-brain", type=int, default=12)
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    narr = _narratives_root()
    os.environ["NARRATIVES_ROOT"] = narr

    if args.models:
        models = list(args.models)
    else:
        models = _models_for_tier("all" if args.tier == "all" else args.tier)
    if not args.models and args.tier in ("large", "all") and not _can_large():
        models = [m for m in models if not LAYER_CONFIG[m].get("requires_gpu")]
        if args.tier == "large":
            print("No GPU / ANIMA_TRAIN_LARGE=1 — nothing to run for large tier.")
            return

    results = []
    for model in models:
        cfg = LAYER_CONFIG[model]
        rec = {"model": model, "family": cfg.get("family"), "hf_id": model}

        if cfg.get("requires_gpu") and not _can_large():
            rec["status"] = "skipped"
            rec["reason"] = "requires_gpu"
            results.append(rec)
            continue
        if cfg.get("gated") and not _hf_authenticated():
            rec["status"] = "skipped"
            rec["reason"] = "gated_hf_login_required"
            results.append(rec)
            continue

        hd = cfg["hidden_dim"]
        max_s = args.max_samples or (100 if hd <= 768 else 70 if hd <= 1024 else 50)
        ep_t = args.epochs_text or (8 if hd <= 2048 else 6)

        if cfg.get("requires_gpu"):
            os.environ["ANIMA_FORCE_CPU"] = "0"
            device = "auto"
        else:
            os.environ["ANIMA_FORCE_CPU"] = "1"
            device = "cpu"

        try:
            print(f"\n### TEXT {model}", flush=True)
            probe_t, meta_t = train_text_probe(model, max_samples=max_s, epochs=ep_t, device=device)
            p_text = save_probe_bundle(probe_t, probe_slug(model), meta_t, suffix="_text")
            rec["text_checkpoint"] = str(p_text)
            rec["text_status"] = "ok"

            if not args.text_only:
                print(f"### BRAIN {model}", flush=True)
                ex = ActivationExtractor(model, device=device)
                _, meta_b = train_probe(
                    ex,
                    narr,
                    probe_slug(model),
                    epochs=args.epochs_brain,
                    target_mode="pca",
                )
                ex.cleanup()
                rec["brain_checkpoint"] = meta_b.get("checkpoint")
                rec["brain_status"] = "ok"
                rec["probe_origin_brain"] = meta_b.get("probe_origin")

            rec["status"] = "ok"
        except Exception as exc:
            rec["status"] = "error"
            rec["message"] = str(exc)
            print(f"FAILED {model}: {exc}", flush=True)

        results.append(rec)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    out = ROOT / "probes" / "zoo" / "train_zoo_full_report.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"Success: {ok}/{len(results)}")


if __name__ == "__main__":
    main()
