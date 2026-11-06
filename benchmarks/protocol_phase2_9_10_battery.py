#!/usr/bin/env python3
"""Phase-2/3 battery toward council 9–10 (CPU on locked/live JSON; no new LM forwards).

Seals:
  - invert-τ sweep for non-saturating OP (0 < k < n) + coverage@risk + err|green
  - length→label vs residualized composite / inv-gap / probe
  - length-matched subset AUROC
  - channel baseline ladder on same held-out fold

Honesty: shipping_claim=false; does not mutate guard_config.yaml.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from benchmarks.guard_metrics import auroc_from_scores
from benchmarks.risk_coverage_and_length_match import length_matched_halueval, risk_coverage
from core.guard import evaluate_guard
from scripts.protocol_r_retune import question_holdout_split

REPORTS = Path(__file__).resolve().parent / "reports"
PAPER = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact")

# Council 9–10 acceptance thresholds (targets, not claims)
TARGETS = {
    "min_heldout_auroc": 0.75,
    "max_error_among_green": 0.20,
    "min_coverage_at_risk_0_10": 0.30,
    "min_coverage_at_risk_0_20": 0.50,
    "require_0_lt_k_lt_n": True,
    "residual_auroc_beats_length": True,  # residual > length→label OR length < 0.6
    "length_to_label_max_for_primary": 0.80,
    "shipping_claim": False,
}


def _row_channels(row: dict[str, Any]) -> dict[str, float]:
    g = evaluate_guard(
        affect=row["affect"],
        uncertainty_signals=row["uncertainty_signals"],
        token_text=row.get("token_text", ""),
    )
    us = row.get("uncertainty_signals") or {}
    aff = row.get("affect") or {}
    # Live rows store uncertainty-scaled logit_gap; inv_logit_gap = 1 - logit_gap
    # (matches Protocol G′). Do not default to 0.5 when key absent.
    if "inv_logit_gap" in us or "inverse_logit_gap" in us:
        inv_gap = float(us.get("inv_logit_gap", us.get("inverse_logit_gap")))
    elif us.get("logit_gap") is not None and us.get("logit_gap") == us.get("logit_gap"):
        inv_gap = 1.0 - float(us["logit_gap"])
    else:
        inv_gap = 0.5
    return {
        "composite": float(g.composite_score),
        "inv_composite": 1.0 - float(g.composite_score),
        "probe": float(aff.get("uncertainty", us.get("probe_uncertainty", 0.5))),
        "fused": float(us.get("fused", us.get("fused_uncertainty", 0.5))),
        "inv_logit_gap": inv_gap,
        "entropy": float(us.get("predictive_entropy", us.get("entropy", 0.0))),
        "answer_length": float(len(str(row.get("answer") or ""))),
    }


def _coverage_at_risk(scores: list[float], labels: list[int], risk: float) -> dict[str, Any]:
    curve = risk_coverage(scores, labels, higher_is_risk=True)
    feas = [
        p
        for p in curve
        if p.get("positive_rate_on_answered") is not None
        and float(p["positive_rate_on_answered"]) <= risk + 1e-9
    ]
    if not feas:
        return {"risk_target": risk, "coverage": 0.0, "status": "infeasible"}
    best = max(feas, key=lambda p: float(p["coverage"]))
    return {
        "risk_target": risk,
        "coverage": best["coverage"],
        "abstain_frac": best["abstain_frac"],
        "positive_rate_on_answered": best["positive_rate_on_answered"],
        "status": "ok",
    }


def _badge_stats(scores: list[float], labels: list[int], tau: float) -> dict[str, Any]:
    abstain = [s >= tau for s in scores]
    green = [not a for a in abstain]
    n = len(scores)
    n_abs = sum(abstain)
    n_green = sum(green)
    n_mis = sum(1 for g, y in zip(green, labels) if g and y == 1)
    return {
        "tau": tau,
        "n": n,
        "n_abstain": n_abs,
        "n_green": n_green,
        "abstain_events": f"{n_abs}/{n}",
        "false_green_rate": round(n_mis / max(1, n), 4),
        "error_among_green": round(n_mis / max(1, n_green), 4) if n_green else None,
        "selective": 0 < n_abs < n,
    }


def _residualize(scores: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    if len(scores) < 3 or np.std(lengths) < 1e-9:
        return scores.copy()
    X = np.column_stack([np.ones(len(lengths)), lengths])
    beta, *_ = np.linalg.lstsq(X, scores, rcond=None)
    return scores - X @ beta


def analyze_slug(slug: str, suite: str, label_key: str, path: Path, seed: int = 42) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
    questions = [str(r.get("question") or "") for r in rows]
    tune_m, test_m = question_holdout_split(questions, seed=seed)
    tune = [r for r, m in zip(rows, tune_m) if m]
    test = [r for r, m in zip(rows, test_m) if m]

    tune_ch = [_row_channels(r) for r in tune]
    test_ch = [_row_channels(r) for r in test]
    y_tune = [int(bool(r[label_key])) for r in tune]
    y_test = [int(bool(r[label_key])) for r in test]

    auroc_tune = auroc_from_scores([c["composite"] for c in tune_ch], y_tune)
    anti = auroc_tune < 0.5
    score_key = "inv_composite" if anti else "composite"

    # Length metrics on held-out
    len_scores = [c["answer_length"] for c in test_ch]
    length_auroc = auroc_from_scores(len_scores, y_test)
    raw = np.array([c[score_key] for c in test_ch], float)
    lengths = np.array(len_scores, float)
    resid = _residualize(raw, lengths)
    resid_auroc = auroc_from_scores(resid.tolist(), y_test)

    channel_aurocs = {
        k: auroc_from_scores([c[k] for c in test_ch], y_test)
        for k in ("composite", "inv_composite", "probe", "fused", "inv_logit_gap", "entropy", "answer_length")
    }

    # Sweep absolute τ on inverted/aligned score
    scores = [c[score_key] for c in test_ch]
    sweep = []
    for tau in [round(x, 2) for x in np.linspace(0.35, 0.85, 26)]:
        badge = _badge_stats(scores, y_test, tau)
        cov10 = _coverage_at_risk(scores, y_test, 0.10)
        cov20 = _coverage_at_risk(scores, y_test, 0.20)
        auroc = auroc_from_scores(scores, y_test)
        hit = (
            badge["selective"]
            and auroc >= TARGETS["min_heldout_auroc"]
            and (badge["error_among_green"] is not None)
            and badge["error_among_green"] <= TARGETS["max_error_among_green"]
            and cov10.get("status") == "ok"
            and float(cov10["coverage"]) >= TARGETS["min_coverage_at_risk_0_10"]
            and float(cov20["coverage"]) >= TARGETS["min_coverage_at_risk_0_20"]
            and resid_auroc > length_auroc
        )
        sweep.append(
            {
                "tau": tau,
                "auroc": auroc,
                "badge": badge,
                "coverage_at_risk_0.10": cov10,
                "coverage_at_risk_0.20": cov20,
                "hits_9_10_targets": hit,
            }
        )

    # Also percentile τ from tune fold
    tune_scores = [c[score_key] for c in tune_ch]
    pct_rows = []
    for pct in (50, 60, 70, 75, 80, 85, 90, 95):
        tau = float(np.percentile(tune_scores, pct))
        badge = _badge_stats(scores, y_test, tau)
        cov10 = _coverage_at_risk(scores, y_test, 0.10)
        cov20 = _coverage_at_risk(scores, y_test, 0.20)
        auroc = auroc_from_scores(scores, y_test)
        pct_rows.append(
            {
                "tune_percentile": pct,
                "tau": round(tau, 4),
                "auroc": auroc,
                "badge": badge,
                "coverage_at_risk_0.10": cov10,
                "coverage_at_risk_0.20": cov20,
                "selective": badge["selective"],
            }
        )

    # Length-matched subset (full set, not just held-out — diagnostic)
    matched = length_matched_halueval(rows, max_len_ratio=1.5) if suite == "halueval" else []
    matched_block: dict[str, Any]
    if len(matched) >= 20:
        mch = [_row_channels(r) for r in matched]
        my = [int(bool(r[label_key])) for r in matched]
        matched_block = {
            "n": len(matched),
            "auroc_inv_or_raw": auroc_from_scores(
                [c[score_key] for c in mch], my
            ),
            "length_auroc": auroc_from_scores([c["answer_length"] for c in mch], my),
            "probe_auroc": auroc_from_scores([c["probe"] for c in mch], my),
            "inv_logit_gap_auroc": auroc_from_scores([c["inv_logit_gap"] for c in mch], my),
        }
    else:
        matched_block = {"n": len(matched), "status": "too_small"}

    hits = [s for s in sweep if s["hits_9_10_targets"]]
    # Best selective by lowest err|green then highest cov@0.20
    selective = [s for s in sweep if s["badge"]["selective"]]
    best_selective = None
    if selective:
        best_selective = sorted(
            selective,
            key=lambda s: (
                s["badge"]["error_among_green"] if s["badge"]["error_among_green"] is not None else 9,
                -float(s["coverage_at_risk_0.20"].get("coverage") or 0),
            ),
        )[0]

    return {
        "status": "ok",
        "model_slug": slug,
        "suite": suite,
        "source": str(path),
        "n_full": len(rows),
        "n_tune": len(tune),
        "n_heldout": len(test),
        "tune_auroc_composite": auroc_tune,
        "anti_aligned": anti,
        "score_key": score_key,
        "heldout_channel_aurocs": channel_aurocs,
        "length_to_label_auroc": length_auroc,
        "residualized_score_auroc": resid_auroc,
        "residual_beats_length": resid_auroc > length_auroc,
        "length_matched": matched_block,
        "absolute_tau_sweep": sweep,
        "percentile_tau_sweep": pct_rows,
        "n_hits_9_10_targets": len(hits),
        "best_selective": best_selective,
        "any_9_10_hit": len(hits) > 0,
    }


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    p.add_argument("--also-paper", action="store_true")
    args = p.parse_args(argv)

    jobs = [
        ("qwen_qwen2_5-0_5b-instruct", "halueval", "is_hallucination"),
        ("qwen_qwen2_5-0_5b-instruct", "truthfulqa", "should_abstain"),
        ("qwen_qwen2_5-7b-instruct", "halueval", "is_hallucination"),
        ("tinyllama_tinyllama-1_1b-chat-v1_0", "halueval", "is_hallucination"),
        ("distilgpt2", "halueval", "is_hallucination"),
    ]

    rollup: dict[str, Any] = {
        "schema": "anima.phase2_9_10_battery.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets_for_council_9_10": TARGETS,
        "shipping_claim": False,
        "cells": {},
        "summary": {},
    }

    for slug, suite, lk in jobs:
        path = args.reports_dir / f"{suite}_live_{slug}.json"
        key = f"{slug}::{suite}"
        if not path.is_file():
            rollup["cells"][key] = {"status": "missing", "path": str(path)}
            print(f"MISSING {key}")
            continue
        block = analyze_slug(slug, suite, lk, path)
        rollup["cells"][key] = block
        best = block.get("best_selective")
        print(
            f"{key}: anti={block['anti_aligned']} len={block['length_to_label_auroc']:.3f} "
            f"resid={block['residualized_score_auroc']:.3f} beats_len={block['residual_beats_length']} "
            f"hits9={block['n_hits_9_10_targets']} "
            f"best_sel={best['badge']['abstain_events'] if best else None} "
            f"errG={best['badge']['error_among_green'] if best else None}"
        )

    any_hit = any(c.get("any_9_10_hit") for c in rollup["cells"].values() if c.get("status") == "ok")
    rollup["summary"] = {
        "any_cell_hits_9_10_targets": any_hit,
        "blocking_gaps": [
            "Need residual AUROC > length→label (or primary suite with length AUROC < 0.8)",
            "Need selective OP with err|green ≤ 0.20 and cov@0.10 ≥ 0.30",
            "Need second trained probe checkpoint (not G′ re-fusion)",
            "Need Protocol G free-gen sealed cell",
            "Need TQA Stage B n≥80",
        ],
        "note": (
            "If any_cell_hits_9_10_targets is false, council 9–10 is not available on "
            "current claim-encode rows with invert/threshold sweeps alone."
        ),
    }

    def _jsonable(o: Any) -> Any:
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    out = args.reports_dir / "protocol_phase2_9_10_battery.json"
    out.write_text(json.dumps(rollup, indent=2, default=_jsonable), encoding="utf-8")
    print(f"Wrote {out}")
    print("SUMMARY any_9_10_hit=", any_hit)

    if args.also_paper:
        PAPER.mkdir(parents=True, exist_ok=True)
        (PAPER / out.name).write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Copied → {PAPER / out.name}")


if __name__ == "__main__":
    main()
