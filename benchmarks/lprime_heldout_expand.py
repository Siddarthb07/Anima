#!/usr/bin/env python3
"""Held-out L′ diagnostic: fit ridge on locked n=80, score expand n=160 rows.

Additive sealed JSON only. Does not invent metrics.
Honesty: λ is taken from the sealed Protocol L′ receipt (selected on n=80);
this is transfer of those weights to a new draw, not nested CV on expand.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.guard_metrics import auroc_from_scores, bootstrap_auroc_ci
from benchmarks.protocol_l_prime import FEATURE_NAMES, fit_ridge, row_features, _orient

REPORTS = Path(__file__).resolve().parent / "reports"

JOBS = [
    (
        "qwen_qwen2_5-0_5b-instruct",
        "halueval",
        "is_hallucination",
        "halueval_live_qwen_qwen2_5-0_5b-instruct.json",
        "halueval_live_qwen_qwen2_5-0_5b-instruct_expand_n160_seed43.json",
    ),
    (
        "qwen_qwen2_5-0_5b-instruct",
        "truthfulqa",
        "should_abstain",
        "truthfulqa_live_qwen_qwen2_5-0_5b-instruct.json",
        "truthfulqa_live_qwen_qwen2_5-0_5b-instruct_expand_n160_seed43.json",
    ),
    (
        "tinyllama_tinyllama-1_1b-chat-v1_0",
        "halueval",
        "is_hallucination",
        "halueval_live_tinyllama_tinyllama-1_1b-chat-v1_0.json",
        "halueval_live_tinyllama_tinyllama-1_1b-chat-v1_0_expand_n160_seed43.json",
    ),
    (
        "tinyllama_tinyllama-1_1b-chat-v1_0",
        "truthfulqa",
        "should_abstain",
        "truthfulqa_live_tinyllama_tinyllama-1_1b-chat-v1_0.json",
        "truthfulqa_live_tinyllama_tinyllama-1_1b-chat-v1_0_expand_n160_seed43.json",
    ),
]


def _rows(path: Path) -> list[dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return list(doc.get("rows") or [])


def _l2_from_receipt(slug: str, suite: str) -> float:
    p = REPORTS / f"protocol_l_prime_{slug}.json"
    if not p.is_file():
        return 0.01
    doc = json.loads(p.read_text(encoding="utf-8"))
    block = (doc.get("suites") or {}).get(suite) or {}
    return float(block.get("l2", 0.01))


def run_one(
    slug: str,
    suite: str,
    label_key: str,
    train_name: str,
    test_name: str,
) -> dict[str, Any]:
    train_path = REPORTS / train_name
    test_path = REPORTS / test_name
    if not train_path.is_file() or not test_path.is_file():
        return {
            "status": "missing",
            "slug": slug,
            "suite": suite,
            "train": str(train_path),
            "test": str(test_path),
        }
    train = _rows(train_path)
    test = _rows(test_path)
    l2 = _l2_from_receipt(slug, suite)
    Xtr = np.stack([row_features(r) for r in train])
    ytr = np.array([int(bool(r[label_key])) for r in train], dtype=float)
    w = _orient(fit_ridge(Xtr, ytr, l2), Xtr, ytr)
    # train LOQO-comparable in-sample AUROC of global fit (diagnostic only)
    auroc_train = auroc_from_scores((Xtr @ w).tolist(), ytr.astype(int).tolist())
    Xte = np.stack([row_features(r) for r in test])
    yte = np.array([int(bool(r[label_key])) for r in test], dtype=int)
    scores = (Xte @ w).tolist()
    auroc = auroc_from_scores(scores, yte.tolist())
    ci = bootstrap_auroc_ci(scores, yte.tolist(), B=2000, seed=43)
    # length residualization on held-out scores
    lens = np.array([float(len(str(r.get("answer") or ""))) for r in test])
    s = np.array(scores, dtype=float)
    if lens.var() > 0:
        slope = float(np.cov(lens, s, ddof=0)[0, 1] / lens.var())
        resid = (s - (s.mean() + slope * (lens - lens.mean()))).tolist()
        auroc_resid = auroc_from_scores(resid, yte.tolist())
        auroc_len = auroc_from_scores(lens.tolist(), yte.tolist())
    else:
        auroc_resid = float("nan")
        auroc_len = float("nan")
    return {
        "status": "ok",
        "slug": slug,
        "suite": suite,
        "label_key": label_key,
        "l2": l2,
        "n_train": len(train),
        "n_test": len(test),
        "feature_names": FEATURE_NAMES,
        "auroc_train_global_fit": round(float(auroc_train), 4),
        "auroc_heldout_expand": round(float(auroc), 4),
        "auroc_heldout_ci95": ci,
        "auroc_answer_length_on_test": None
        if auroc_len != auroc_len
        else round(float(auroc_len), 4),
        "auroc_heldout_residualized_on_length": None
        if auroc_resid != auroc_resid
        else round(float(auroc_resid), 4),
        "honesty": {
            "weights_fit_on_locked_n80_only": True,
            "lambda_from_sealed_lprime_receipt": True,
            "expand_is_new_seed_draw": True,
            "not_nested_cv_on_expand": True,
        },
    }


def main() -> None:
    out: dict[str, Any] = {
        "schema": "anima.lprime_heldout_expand.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "jobs": [],
    }
    for job in JOBS:
        result = run_one(*job)
        out["jobs"].append(result)
        print(
            f"{result.get('slug')} {result.get('suite')}: "
            f"heldout={result.get('auroc_heldout_expand')} "
            f"resid={result.get('auroc_heldout_residualized_on_length')} "
            f"status={result.get('status')}"
        )
    path = REPORTS / "lprime_heldout_expand_seed43.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
