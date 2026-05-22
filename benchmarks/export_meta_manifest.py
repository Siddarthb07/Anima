"""Build a benchmark manifest from committed *.meta.json (no HF model load)."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.run_halueval_guard import run as run_halu
from benchmarks.run_truthfulqa_guard import run as run_tqa
from probes.zoo_io import load_meta, meta_path, probe_slug

ROOT = Path(__file__).resolve().parent.parent


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def export(model: str) -> None:
    slug = probe_slug(model)
    entries: list[dict] = []

    text_meta = load_meta(slug, "_text")
    if meta_path(slug, "_text").exists():
        entries.append(
            {
                "tier": "external_text",
                "benchmark": "go_emotions",
                "status": "ok",
                "source": "train_text.meta.json",
                "pearson_valence": text_meta.get("val_pearson_valence"),
                "pearson_arousal": text_meta.get("val_pearson_arousal"),
                "n_samples": text_meta.get("n_samples"),
                "probe_origin": text_meta.get("probe_origin"),
            }
        )

    brain_meta = load_meta(slug, "_narratives_pca")
    if meta_path(slug, "_narratives_pca").exists():
        entries.append(
            {
                "tier": "external",
                "benchmark": "narratives_holdout",
                "status": "ok",
                "source": "narratives_pca.meta.json",
                "val_mse": brain_meta.get("val_mse"),
                "val_r_valence": brain_meta.get("val_r_valence"),
                "val_r_arousal": brain_meta.get("val_r_arousal"),
                "holdout_stories": brain_meta.get("holdout_stories"),
                "probe_origin": brain_meta.get("probe_origin"),
                "baselines": brain_meta.get("baselines"),
            }
        )
        entries.append(
            {
                "tier": "external",
                "benchmark": "litcoder_style_ridge",
                "status": "ok",
                "source": "narratives_pca.meta.json",
                "val_mse": brain_meta.get("val_mse"),
                "val_r_valence": brain_meta.get("val_r_valence"),
                "val_r_arousal": brain_meta.get("val_r_arousal"),
                "holdout_word_rate_mean_r": (brain_meta.get("baselines") or {})
                .get("holdout_lucy", {})
                .get("word_rate_mean_r"),
            }
        )

    entries.append(run_halu(model))
    entries.append(run_tqa(model))

    reports = ROOT / "benchmarks" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": model,
        "git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    stamped = reports / f"{model.replace('/', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_meta"
    stamped.mkdir(parents=True, exist_ok=True)
    path = stamped / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    out = reports / f"latest_{slug}_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    print(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilgpt2")
    args = p.parse_args()
    export(args.model)


if __name__ == "__main__":
    main()
