"""Compare Anima ridge baseline to encoding_pipeline word-rate baseline on Narratives."""

from __future__ import annotations

import os
from pathlib import Path


def run(model: str) -> dict:
    root = os.environ.get("NARRATIVES_ROOT", "")
    if not root or not Path(root).exists():
        return {"tier": "external", "benchmark": "litcoder_style_ridge", "status": "skipped"}
    try:
        from benchmarks.run_narratives_encoding import run as narr_run

        out = narr_run(model)
        out["benchmark"] = "litcoder_style_ridge"
        baselines = out.get("baselines", {})
        holdout_key = [k for k in baselines if k.startswith("holdout_")]
        wr = [baselines[k].get("word_rate_mean_r", 0) for k in holdout_key if isinstance(baselines[k], dict)]
        out["holdout_word_rate_mean_r"] = round(sum(wr) / len(wr), 4) if wr else None
        return out
    except Exception as exc:
        return {"tier": "external", "benchmark": "litcoder_style_ridge", "status": "error", "message": str(exc)}
