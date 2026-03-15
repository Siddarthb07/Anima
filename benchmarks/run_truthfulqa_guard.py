"""TruthfulQA guard benchmark — scores abstain policy on fixture or custom JSON."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from benchmarks.guard_metrics import _default_truthfulqa_path, load_fixture, score_abstain_accuracy


def run(model: str) -> dict[str, Any]:
    path = os.environ.get("TRUTHFULQA_PATH", "")
    if not path:
        path = str(_default_truthfulqa_path())
    if not Path(path).is_file():
        return {
            "benchmark": "truthfulqa_guard",
            "tier": "external_guard",
            "status": "skipped",
            "reason": "Set TRUTHFULQA_PATH or add benchmarks/fixtures/truthfulqa_guard_sample.json",
            "model": model,
        }
    rows = load_fixture(path)
    metrics = score_abstain_accuracy(rows, label_key="should_abstain")
    return {
        "benchmark": "truthfulqa_guard",
        "tier": "external_guard",
        "status": "ok",
        "model": model,
        "fixture": path,
        "note": "Scores guard policy on supplied readout rows, not LM factuality",
        **metrics,
    }
