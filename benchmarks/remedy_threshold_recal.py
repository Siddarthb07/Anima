#!/usr/bin/env python3
"""Learn-then-Test / conformal-style τ recalibration remedy (CPU demo).

Given claim-encode scores + binary hall labels:
  1. Split tune/cal/test by question holdout.
  2. On cal, pick the largest coverage τ such that empirical err|green ≤ α
     (CRC-style risk control on the answered set), or fall back to quantile.
  3. Report occupancy gap (tune-as-fixture vs test) and Cov@risk before/after.

This is a remedy *sketch* for the paper: shows that threshold recal alone can
move occupancy without inventing a new detector. shipping_claim=false.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from benchmarks.guard_metrics import auroc_from_scores
from benchmarks.occupancy_ks_bound import evaluate_cell, fixture_quantile_tau
from benchmarks.protocol_phase2_9_10_battery import (
    _badge_stats,
    _coverage_at_risk,
    _row_channels,
)

REPORTS = Path(__file__).resolve().parent / "reports"
PAPER_ART = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact\prereg")


def learn_then_test_tau(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    alpha: float = 0.20,
    grid: int = 101,
) -> dict[str, Any]:
    """Largest coverage operating point with err|green ≤ α on calibration."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    lo, hi = float(np.min(s)), float(np.max(s))
    if not np.isfinite(lo) or hi <= lo:
        return {"tau": float(np.median(s)), "status": "degenerate", "coverage": 0.0}
    candidates = []
    for t in np.linspace(lo, hi, num=grid):
        badge = _badge_stats(list(s), list(y), float(t))
        err = badge["error_among_green"]
        if err is None:
            continue
        if err <= alpha + 1e-12:
            cov = badge["n_green"] / max(1, badge["n"])
            candidates.append((cov, float(t), err, badge["n_abstain"]))
    if not candidates:
        # Fall back: quantile that abstains ~α mass of high scores
        tau = fixture_quantile_tau(s, q=1.0 - alpha)
        return {"tau": tau, "status": "fallback_quantile", "coverage": None, "alpha": alpha}
    cov, tau, err, n_abs = max(candidates, key=lambda x: (x[0], -x[1]))
    return {
        "tau": tau,
        "status": "crc_feasible",
        "coverage": cov,
        "err_among_green": err,
        "n_abstain": n_abs,
        "alpha": alpha,
    }


def run_demo(
    path: Path,
    *,
    channel: str = "inv_composite",
    alpha: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    from scripts.protocol_r_retune import question_holdout_split

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    questions = [str(r.get("question") or "") for r in rows]
    tune_m, test_m = question_holdout_split(questions, seed=seed)
    # Use half of tune as cal
    tune_idx = [i for i, m in enumerate(tune_m) if m]
    rng = np.random.default_rng(seed)
    rng.shuffle(tune_idx)
    mid = len(tune_idx) // 2
    cal_set = set(tune_idx[:mid])
    learn_set = set(tune_idx[mid:])
    test_set = {i for i, m in enumerate(test_m) if m}

    def pack(idxs: set[int]) -> tuple[list[float], list[int]]:
        ch = [_row_channels(rows[i]) for i in sorted(idxs)]
        y = [
            int(bool(rows[i].get("is_hallucination", rows[i].get("hallucinated", False))))
            for i in sorted(idxs)
        ]
        return [c[channel] for c in ch], y

    s_learn, y_learn = pack(learn_set)
    s_cal, y_cal = pack(cal_set)
    s_test, y_test = pack(test_set)

    tau_before = fixture_quantile_tau(s_learn, q=0.90)
    ltt = learn_then_test_tau(s_cal, y_cal, alpha=alpha)
    tau_after = float(ltt["tau"])

    before = {
        "tau": tau_before,
        "badge": _badge_stats(s_test, y_test, tau_before),
        "cov_at_0_10": _coverage_at_risk(s_test, y_test, 0.10),
        "auroc": auroc_from_scores(s_test, y_test),
    }
    after = {
        "tau": tau_after,
        "badge": _badge_stats(s_test, y_test, tau_after),
        "cov_at_0_10": _coverage_at_risk(s_test, y_test, 0.10),
        "auroc": auroc_from_scores(s_test, y_test),
        "ltt": ltt,
    }
    # Occupancy: learn scores as fixture proxy vs test
    occ_before = evaluate_cell("before", s_learn, s_test, tau_before, channel=channel)
    occ_after = evaluate_cell("after", s_learn, s_test, tau_after, channel=channel)

    def _jsonify(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    return _jsonify(
        {
            "schema": "anima.remedy_threshold_recal.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": str(path),
            "channel": channel,
            "alpha": alpha,
            "seed": seed,
            "n_learn": len(s_learn),
            "n_cal": len(s_cal),
            "n_test": len(s_test),
            "shipping_claim": False,
            "before": before,
            "after": after,
            "occupancy": {
                "before": {
                    "gap": occ_before.occupancy_gap,
                    "ks": occ_before.ks,
                    "bound_holds": occ_before.bound_holds,
                    "live_abstain": occ_before.live_abstain_rate,
                },
                "after": {
                    "gap": occ_after.occupancy_gap,
                    "ks": occ_after.ks,
                    "bound_holds": occ_after.bound_holds,
                    "live_abstain": occ_after.live_abstain_rate,
                },
            },
            "notes": "CPU LtT/CRC sketch; not a shipping unlock. shipping_claim=false.",
        }
    )


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--live",
        type=Path,
        default=REPORTS / "halueval_live_qwen_qwen2_5-0_5b-instruct_lenbal_n300.json",
    )
    p.add_argument("--out", type=Path, default=REPORTS / "remedy_threshold_recal_demo.json")
    p.add_argument("--channel", default="inv_composite")
    p.add_argument("--alpha", type=float, default=0.20)
    p.add_argument("--also-paper", action="store_true")
    args = p.parse_args(argv)

    if not args.live.exists():
        raise SystemExit(f"missing {args.live}")
    demo = run_demo(args.live, channel=args.channel, alpha=args.alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(demo, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        f"  before τ={demo['before']['tau']:.4f} abs={demo['before']['badge']['abstain_events']} "
        f"errG={demo['before']['badge']['error_among_green']}"
    )
    print(
        f"  after  τ={demo['after']['tau']:.4f} abs={demo['after']['badge']['abstain_events']} "
        f"errG={demo['after']['badge']['error_among_green']} status={demo['after']['ltt']['status']}"
    )
    print(
        f"  occ gap {demo['occupancy']['before']['gap']:.4f} → {demo['occupancy']['after']['gap']:.4f} "
        f"(KS={demo['occupancy']['before']['ks']:.4f})"
    )
    if args.also_paper:
        PAPER_ART.mkdir(parents=True, exist_ok=True)
        dest = PAPER_ART / args.out.name
        dest.write_text(json.dumps(demo, indent=2), encoding="utf-8")
        print(f"also {dest}")


if __name__ == "__main__":
    main()
