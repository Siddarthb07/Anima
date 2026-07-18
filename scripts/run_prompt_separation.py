"""Live prompt separation: positive vs negative mean valence (chat-template aware)."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from core.extractor import ActivationExtractor
from probes.linear_probe import AffectProbe
from probes.zoo_io import load_probe_into

REPORTS = Path(__file__).resolve().parent.parent / "benchmarks" / "reports"

DEFAULT_PROMPTS = {
    "positive": [
        "I am so happy and grateful for today.",
        "This is wonderful news — I feel joyful and excited.",
        "I love spending time with my family; everything feels warm and good.",
    ],
    "negative": [
        "I feel devastated and hopeless about everything.",
        "This is terrible; I am angry and deeply sad.",
        "I hate this situation and feel empty and afraid.",
    ],
}


def mean_valence(model_name: str, prompts: list[str], *, device: str = "cpu") -> dict[str, Any]:
    os.environ.setdefault("ANIMA_FORCE_CPU", "1")
    os.environ.setdefault("ANIMA_PREFER_TEXT_PROBE", "1")
    extractor = ActivationExtractor(model_name, device=device)
    probe = AffectProbe(extractor.hidden_dim, len(extractor.layer_indices))
    meta = load_probe_into(probe, model_name, map_location="cpu")
    vals: list[float] = []
    for p in prompts:
        rows = extractor.encode_sequence(p, max_length=128)
        if not rows:
            continue
        with torch.no_grad():
            out = probe(rows[-1]["activations"])
        vals.append(float(out["valence"]))
    extractor.cleanup()
    mean = sum(vals) / max(1, len(vals))
    return {
        "n": len(vals),
        "mean_valence": round(mean, 4),
        "values": [round(v, 4) for v in vals],
        "probe_origin": meta.get("probe_origin"),
        "use_chat_template": meta.get("use_chat_template"),
    }


def run(model_name: str, *, device: str = "cpu") -> dict[str, Any]:
    pos = mean_valence(model_name, DEFAULT_PROMPTS["positive"], device=device)
    neg = mean_valence(model_name, DEFAULT_PROMPTS["negative"], device=device)
    gap = round(pos["mean_valence"] - neg["mean_valence"], 4)
    report = {
        "model": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "positive": pos,
        "negative": neg,
        "pos_neg_gap": gap,
        "targets": {
            "pos_mean_ge": 0.25,
            "neg_mean_le": 0.05,
            "gap_ge": 0.20,
        },
        "passed": bool(
            pos["mean_valence"] >= 0.25 and neg["mean_valence"] <= 0.05 and gap >= 0.20
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    slug = model_name.replace("/", "_").replace(".", "_").lower()
    path = REPORTS / f"prompt_separation_{slug}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {path}")
    return report


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    run(args.model, device=args.device)


if __name__ == "__main__":
    main()
