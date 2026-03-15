"""Shared guard benchmark scoring from JSON fixtures or live readouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.guard import evaluate_guard

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _default_truthfulqa_path() -> Path:
    return FIXTURES / "truthfulqa_guard_sample.json"


def _default_halueval_path() -> Path:
    return FIXTURES / "halueval_guard_sample.json"


def load_fixture(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def score_abstain_accuracy(rows: list[dict[str, Any]], *, label_key: str) -> dict[str, float]:
    """Binary label vs guard abstain_recommended."""
    tp = fp = tn = fn = 0
    scores: list[float] = []
    labels: list[int] = []
    for row in rows:
        g = evaluate_guard(
            affect=row["affect"],
            uncertainty_signals=row["uncertainty_signals"],
            token_text=row.get("token_text", ""),
        )
        pred = int(g.abstain_recommended)
        y = int(bool(row[label_key]))
        scores.append(g.composite_score)
        labels.append(y)
        if pred and y:
            tp += 1
        elif pred and not y:
            fp += 1
        elif not pred and y:
            fn += 1
        else:
            tn += 1
    n = max(1, len(rows))
    acc = (tp + tn) / n
    # AUROC via rank if both classes present
    auroc = 0.5
    if len(set(labels)) == 2:
        pos = [s for s, y in zip(scores, labels) if y == 1]
        neg = [s for s, y in zip(scores, labels) if y == 0]
        if pos and neg:
            wins = sum(1 for p in pos for n in neg if p > n)
            ties = sum(1 for p in pos for n in neg if p == n)
            auroc = (wins + 0.5 * ties) / (len(pos) * len(neg))
    return {
        "abstain_accuracy": round(acc, 4),
        "auroc_composite": round(auroc, 4),
        "n_samples": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
