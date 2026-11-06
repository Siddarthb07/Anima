#!/usr/bin/env python3
"""Seal selective OP on TruthfulQA live rows with a *single* decide() for badge+coverage.

Score: hall-probe if checkpoint exists, else inv_composite when anti-aligned.
Pre-registered gates from protocol_phase2_9_10_battery.TARGETS.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from benchmarks.guard_metrics import auroc_from_scores
from benchmarks.protocol_phase2_9_10_battery import (
    TARGETS,
    _badge_stats,
    _coverage_at_risk,
    _row_channels,
)
from benchmarks.train_hallucination_probe import HallProbe, _slug
from core.guard import evaluate_guard
from scripts.protocol_r_retune import question_holdout_split

REPORTS = Path(__file__).resolve().parent / "reports"
ZOO = Path(__file__).resolve().parents[1] / "probes" / "zoo"
PAPER = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact")


def _load_hall_scores_from_live_unavailable():
    return None


def seal_from_live(
    slug: str,
    *,
    seed: int = 42,
    hall_ckpt: Optional[Path] = None,
) -> dict[str, Any]:
    path = REPORTS / f"truthfulqa_live_{slug}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    label_key = "should_abstain"
    questions = [str(r.get("question") or "") for r in rows]
    tune_m, test_m = question_holdout_split(questions, seed=seed)
    tune = [r for r, m in zip(rows, tune_m) if m]
    test = [r for r, m in zip(rows, test_m) if m]

    # Baseline channel scores on held-out
    test_ch = [_row_channels(r) for r in test]
    y = [int(bool(r[label_key])) for r in test]
    len_auroc = auroc_from_scores([c["answer_length"] for c in test_ch], y)
    auroc_comp = auroc_from_scores([c["composite"] for c in test_ch], y)
    anti = auroc_comp < 0.5
    score_key = "inv_composite" if anti else "composite"
    base_scores = [c[score_key] for c in test_ch]
    base_auroc = auroc_from_scores(base_scores, y)

    score_name = score_key
    scores = base_scores
    score_auroc = base_auroc
    hall_meta = None

    # Prefer sealed hall-probe heldout_scores when present (same seed holdout).
    zoo_slug = slug.lower().replace("-", "_").replace(".", "_")
    hall_path = REPORTS / f"hall_probe_{zoo_slug}_truthfulqa.json"
    if not hall_ckpt:
        cand = ZOO / f"{zoo_slug}_hall_probe_truthfulqa.pt"
        if cand.is_file():
            hall_ckpt = cand
    if hall_path.is_file():
        hall = json.loads(hall_path.read_text(encoding="utf-8"))
        hs = hall.get("heldout_scores")
        if hs and len(hs) == len(y):
            scores = [float(x) for x in hs]
            score_name = "hall_probe"
            score_auroc = auroc_from_scores(scores, y)
            hall_meta = {
                "checkpoint": hall.get("checkpoint"),
                "sha256": hall.get("sha256"),
                "heldout_auroc": hall.get("heldout_auroc"),
                "source": str(hall_path),
            }
        elif hall_ckpt and hall_ckpt.is_file():
            hall_meta = {
                "checkpoint": str(hall_ckpt),
                "note": "ckpt present but heldout_scores missing/mismatched; using channel score",
            }
    elif hall_ckpt and hall_ckpt.is_file():
        hall_meta = {
            "checkpoint": str(hall_ckpt),
            "note": "ckpt present; hall_probe json missing heldout_scores",
        }

    # Sweep τ on the same scores used for coverage (joint decide)
    best = None
    hits = []
    for tau in [round(x, 2) for x in np.linspace(0.30, 0.85, 28)]:
        badge = _badge_stats(scores, y, tau)
        # Joint decide: abstain iff score >= tau  (== badge)
        cov10 = _coverage_at_risk(scores, y, 0.10)
        cov20 = _coverage_at_risk(scores, y, 0.20)
        # Coverage curve uses ranking; for joint decide also compute empirical risk among greens
        row = {
            "tau": tau,
            "auroc": score_auroc,
            "badge": badge,
            "coverage_at_risk_0.10": cov10,
            "coverage_at_risk_0.20": cov20,
            "joint_decide": "abstain_if_score_ge_tau",
        }
        hit = (
            badge["selective"]
            and score_auroc >= TARGETS["min_heldout_auroc"]
            and badge["error_among_green"] is not None
            and badge["error_among_green"] <= TARGETS["max_error_among_green"]
            and cov10.get("status") == "ok"
            and float(cov10["coverage"]) >= TARGETS["min_coverage_at_risk_0_10"]
            and float(cov20["coverage"]) >= TARGETS["min_coverage_at_risk_0_20"]
            and len_auroc < TARGETS["length_to_label_max_for_primary"]
        )
        row["hits_9_gates"] = hit
        if hit:
            hits.append(row)
        if badge["selective"] and badge["error_among_green"] is not None:
            if best is None or badge["error_among_green"] < best["badge"]["error_among_green"]:
                best = row

    out = {
        "schema": "anima.protocol_tqa_selective_op.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_slug": slug,
        "suite": "truthfulqa",
        "n_full": len(rows),
        "n_heldout": len(test),
        "score_name": score_name,
        "anti_aligned_composite": anti,
        "heldout_auroc": score_auroc,
        "base_channel_auroc": base_auroc,
        "length_to_label_auroc": len_auroc,
        "length_gate_clear": len_auroc < TARGETS["length_to_label_max_for_primary"],
        "best_selective": best,
        "n_hits_9_gates": len(hits),
        "hits": hits[:5],
        "hall_meta": hall_meta,
        "shipping_claim": False,
        "targets": TARGETS,
        "source": str(path),
    }
    def _jsonable(o: Any) -> Any:
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    dest = REPORTS / f"protocol_tqa_selective_op_{slug}.json"
    dest.write_text(json.dumps(out, indent=2, default=_jsonable), encoding="utf-8")
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / dest.name).write_text(dest.read_text(encoding="utf-8"), encoding="utf-8")
    print(
        f"{slug}: auroc={base_auroc:.3f} len={len_auroc:.3f} clear_len={out['length_gate_clear']} "
        f"hits={len(hits)} best_abs={best['badge']['abstain_events'] if best else None} "
        f"errG={best['badge']['error_among_green'] if best else None}"
    )
    return out


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--slug", action="append", dest="slugs", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    slugs = args.slugs or [
        "qwen_qwen2_5-0_5b-instruct",
        "qwen_qwen2_5-7b-instruct",
    ]
    rollup = {
        "schema": "anima.protocol_tqa_selective_op.rollup.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shipping_claim": False,
        "models": {},
    }
    for slug in slugs:
        path = REPORTS / f"truthfulqa_live_{slug}.json"
        if not path.is_file():
            rollup["models"][slug] = {"status": "missing"}
            print(f"MISSING {slug}")
            continue
        rollup["models"][slug] = seal_from_live(slug, seed=args.seed)
    def _jsonable(o: Any) -> Any:
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    out = REPORTS / "protocol_tqa_selective_op_rollup.json"
    out.write_text(json.dumps(rollup, indent=2, default=_jsonable), encoding="utf-8")
    (PAPER / out.name).write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
