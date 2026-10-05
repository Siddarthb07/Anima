"""Shared guard benchmark scoring from JSON fixtures or live readouts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional, Sequence

from core.guard import evaluate_guard

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _default_truthfulqa_path() -> Path:
    return FIXTURES / "truthfulqa_guard_sample.json"


def _default_halueval_path() -> Path:
    return FIXTURES / "halueval_guard_sample.json"


def load_fixture(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def auroc_from_scores(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank AUROC for arbitrary continuous scores vs binary labels."""
    if len(set(labels)) != 2 or not scores:
        return 0.5
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return round((wins + 0.5 * ties) / (len(pos) * len(neg)), 4)


def bootstrap_auroc_ci(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    B: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Bootstrap 95% CI for Mann–Whitney AUROC (B resamples with replacement).

    Matches locked paper protocol: seed=42, B=2000, pairwise rank AUROC.
    """
    scores_l = list(scores)
    labels_l = [int(y) for y in labels]
    n = len(scores_l)
    point = auroc_from_scores(scores_l, labels_l)
    if n < 2 or len(set(labels_l)) != 2:
        return {
            "point": point,
            "lo": point,
            "hi": point,
            "B": B,
            "seed": seed,
            "n": n,
            "method": "bootstrap_mann_whitney",
        }
    rng = random.Random(seed)
    vals: list[float] = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(
            auroc_from_scores(
                [scores_l[i] for i in idx],
                [labels_l[i] for i in idx],
            )
        )
    vals.sort()
    lo = vals[int(0.025 * B)]
    hi = vals[min(B - 1, int(0.975 * B))]
    return {
        "point": point,
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "B": B,
        "seed": seed,
        "n": n,
        "method": "bootstrap_mann_whitney",
    }


def score_abstain_accuracy(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    disabled_signals: Optional[Sequence[str]] = None,
    with_ci: bool = False,
    ci_B: int = 2000,
    ci_seed: int = 42,
) -> dict[str, Any]:
    """Binary label vs guard abstain_recommended."""
    tp = fp = tn = fn = 0
    scores: list[float] = []
    labels: list[int] = []
    for row in rows:
        g = evaluate_guard(
            affect=row["affect"],
            uncertainty_signals=row["uncertainty_signals"],
            token_text=row.get("token_text", ""),
            disabled_signals=disabled_signals,
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
    auroc = auroc_from_scores(scores, labels)
    out: dict[str, Any] = {
        "abstain_accuracy": round(acc, 4),
        "auroc_composite": auroc,
        "n_samples": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "never_abstains": bool(tp == 0 and fp == 0),
    }
    if with_ci:
        out["auroc_ci95"] = bootstrap_auroc_ci(scores, labels, B=ci_B, seed=ci_seed)
    return out
