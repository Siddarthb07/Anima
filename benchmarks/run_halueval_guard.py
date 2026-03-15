"""HaluEval-style guard benchmark — AUROC on fixture or HALUEVAL_PATH JSON."""

from __future__ import annotations

import os
from pathlib import Path

from benchmarks.guard_metrics import _default_halueval_path, load_fixture, score_abstain_accuracy


def run(model: str) -> dict:
    if os.environ.get("SKIP_HALU_EVAL") == "1":
        return {"tier": "external_guard", "benchmark": "halueval", "status": "skipped"}
    path = os.environ.get("HALUEVAL_PATH", "")
    if not path:
        path = str(_default_halueval_path())
    if not Path(path).is_file():
        return {
            "tier": "external_guard",
            "benchmark": "halueval",
            "status": "skipped",
            "reason": "Set HALUEVAL_PATH or add benchmarks/fixtures/halueval_guard_sample.json",
        }
    rows = load_fixture(path)
    metrics = score_abstain_accuracy(rows, label_key="is_hallucination")
    return {
        "tier": "external_guard",
        "benchmark": "halueval",
        "status": "ok",
        "model": model,
        "fixture": path,
        **metrics,
    }
