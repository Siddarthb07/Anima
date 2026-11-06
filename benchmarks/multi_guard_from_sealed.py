#!/usr/bin/env python3
"""Offline multi-guard stats from sealed claim-encode JSON (no GPU / no HF).

Companion to multi_guard_claim_encode.py (live re-encode). Uses channels already
present on sealed rows: composite, inv_composite, entropy, inv_logit_gap, probe,
fused. DeLong + Holm vs composite; KS fingerprint; 3 seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from benchmarks.guard_metrics import auroc_from_scores
from benchmarks.occupancy_ks_bound import evaluate_cell, fixture_quantile_tau
from benchmarks.occupancy_tau_path import multi_guard_ks_fingerprint
from benchmarks.protocol_phase2_9_10_battery import (
    _badge_stats,
    _coverage_at_risk,
    _row_channels,
)

REPORTS = Path(__file__).resolve().parent / "reports"
PAPER_ART = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact\prereg")
GUARDS = ("composite", "inv_composite", "entropy", "inv_logit_gap", "probe", "fused")


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(abs(z) / math.sqrt(2.0))


def delong_pvalue(
    scores_a: Sequence[float], scores_b: Sequence[float], labels: Sequence[int]
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 < 2 or n0 < 2:
        return {"p": float("nan"), "z": float("nan"), "method": "insufficient_n"}

    def structural(scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        sp, sn = scores[pos], scores[neg]
        v10 = np.array(
            [(np.sum(si > sn) + 0.5 * np.sum(si == sn)) / n0 for si in sp], dtype=float
        )
        v01 = np.array(
            [(np.sum(sp > sj) + 0.5 * np.sum(sp == sj)) / n1 for sj in sn], dtype=float
        )
        return float(np.mean(v10)), v10, v01

    t_a, v10a, v01a = structural(a)
    t_b, v10b, v01b = structural(b)
    var = float(
        np.var(v10a - v10b, ddof=1) / n1 + np.var(v01a - v01b, ddof=1) / n0
    )
    if var <= 0 or not np.isfinite(var):
        return {
            "auroc_a": t_a,
            "auroc_b": t_b,
            "delta": t_a - t_b,
            "p": 1.0,
            "z": 0.0,
            "method": "degenerate",
        }
    z = (t_a - t_b) / math.sqrt(var)
    return {
        "auroc_a": t_a,
        "auroc_b": t_b,
        "delta": t_a - t_b,
        "z": float(z),
        "p": float(2.0 * _norm_sf(z)),
        "method": "delong_approx_structural",
    }


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(
        range(m),
        key=lambda i: (float("inf") if not np.isfinite(pvals[i]) else pvals[i]),
    )
    adj, running = [1.0] * m, 0.0
    for rank, i in enumerate(order):
        p = pvals[i]
        if not np.isfinite(p):
            adj[i] = float("nan")
            continue
        running = max(running, min(1.0, (m - rank) * p))
        adj[i] = running
    return adj


def analyze_live(
    path: Path, *, q_tau: float = 0.90, seeds: Sequence[int] = (41, 42, 43)
) -> dict[str, Any]:
    from scripts.protocol_r_retune import question_holdout_split

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    questions = [str(r.get("question") or "") for r in rows]
    seed_rows: list[dict[str, Any]] = []
    for sd in seeds:
        tune_m, test_m = question_holdout_split(questions, seed=int(sd))
        test = [r for r, m in zip(rows, test_m) if m]
        tune = [r for r, m in zip(rows, tune_m) if m]
        test_ch = [_row_channels(r) for r in test]
        tune_ch = [_row_channels(r) for r in tune]
        y_test = [
            int(bool(r.get("is_hallucination", r.get("hallucinated", False))))
            for r in test
        ]
        guards_out: dict[str, Any] = {}
        for g in GUARDS:
            tune_s = [c[g] for c in tune_ch]
            test_s = [c[g] for c in test_ch]
            arr = np.asarray(test_s, dtype=float)
            finite = arr[np.isfinite(arr)]
            if len(finite) < 10 or float(np.std(finite)) < 1e-12:
                guards_out[g] = {
                    "auroc": None,
                    "skipped": "constant_or_nonfinite_channel",
                    "n_finite": int(len(finite)),
                }
                continue
            test_s = [float(x) if np.isfinite(x) else float("nan") for x in arr]
            tau = fixture_quantile_tau(tune_s, q=q_tau)
            guards_out[g] = {
                "auroc": auroc_from_scores(test_s, y_test),
                "tau_q": tau,
                "badge": _badge_stats(test_s, y_test, tau),
                "cov_at_0_10": _coverage_at_risk(test_s, y_test, 0.10),
            }
        comp_s = [c["composite"] for c in test_ch]
        comparisons = []
        for g in GUARDS:
            if g == "composite" or guards_out.get(g, {}).get("skipped"):
                continue
            comparisons.append(
                {"guard": g, **delong_pvalue(comp_s, [c[g] for c in test_ch], y_test)}
            )
        for c, hp in zip(comparisons, holm_adjust([c["p"] for c in comparisons])):
            c["p_holm"] = hp
        fix_map = {g: [c[g] for c in tune_ch] for g in GUARDS}
        live_map = {g: [c[g] for c in test_ch] for g in GUARDS}
        occ = {}
        for g in GUARDS:
            if guards_out.get(g, {}).get("skipped"):
                continue
            fs, ls = fix_map[g], live_map[g]
            tau = fixture_quantile_tau(fs, q=q_tau)
            cell = evaluate_cell(f"{g}_tune_vs_test", fs, ls, tau, channel=g)
            occ[g] = {
                "ks": cell.ks,
                "occupancy_gap": cell.occupancy_gap,
                "bound_holds": cell.bound_holds,
                "live_abstain": cell.live_abstain_rate,
            }
        seed_rows.append(
            {
                "seed": int(sd),
                "n_tune": len(tune),
                "n_test": len(test),
                "guards": guards_out,
                "delong_vs_composite": comparisons,
                "occupancy_tune_vs_test": occ,
                "ks_fingerprint": multi_guard_ks_fingerprint(fix_map, live_map),
            }
        )
    return {
        "schema": "anima.multi_guard_from_sealed.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(path),
        "shipping_claim": False,
        "guards": list(GUARDS),
        "seeds": seed_rows,
    }


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--live",
        type=Path,
        default=REPORTS / "halueval_live_qwen_qwen2_5-0_5b-instruct_lenbal_n300.json",
    )
    p.add_argument(
        "--out", type=Path, default=REPORTS / "multi_guard_from_sealed_rollup.json"
    )
    p.add_argument("--also-paper", action="store_true")
    args = p.parse_args(argv)
    rollup = analyze_live(args.live)
    args.out.write_text(json.dumps(rollup, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    s0 = rollup["seeds"][0]
    for g, info in s0["guards"].items():
        if info.get("skipped"):
            print(f"  {g}: skipped ({info['skipped']})")
        else:
            print(
                f"  {g}: AUROC={info['auroc']} abs={info['badge']['abstain_events']} "
                f"errG={info['badge']['error_among_green']}"
            )
    print(f"  KS L2={s0['ks_fingerprint']['ks_l2']} dom={s0['ks_fingerprint']['dominant_guard']}")
    if args.also_paper:
        PAPER_ART.mkdir(parents=True, exist_ok=True)
        (PAPER_ART / args.out.name).write_text(json.dumps(rollup, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
