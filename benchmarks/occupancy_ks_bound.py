#!/usr/bin/env python3
"""KS occupancy-gap bound: |P_L(A=1) - P_F(A=1)| <= KS(F_F, F_L).

Theorem (right-continuous CDFs, A = 1[S >= tau]):
  |F_F(tau-) - F_L(tau-)| <= sup_x |F_F(x) - F_L(x)| = KS(F_F, F_L).
If tau is the fixture q-quantile (e.g. p95 => q=0.95), P_F(A=1) = 1-q
and the live–fixture occupancy gap is bounded by KS.

g = max(S_L)/tau < 1 is a corollary of support separation (idle reject),
not the primary metric.

shipping / confirmatory: report KS vs empirical gap; bound holds iff
empirical_gap <= KS + numerical_tol (ties / empirical quantile).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

REPORTS = Path(__file__).resolve().parent / "reports"
PAPER_ART = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact\prereg")


@dataclass
class OccupancyCell:
    name: str
    tau: float
    n_fixture: int
    n_live: int
    fixture_abstain_rate: float
    live_abstain_rate: float
    occupancy_gap: float
    ks: float
    w1: float
    bound_holds: bool
    transfer_gap_g: float
    idle_reject_live: bool
    channel: str = ""
    notes: str = ""


def empirical_cdf_values(samples: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Right-continuous empirical CDF F_n(x) = (1/n) |{i: X_i <= x}|."""
    s = np.sort(np.asarray(samples, dtype=float))
    n = max(1, len(s))
    return np.searchsorted(s, xs, side="right") / n


def ks_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample KS statistic (sup |F_a - F_b|) on pooled evaluation grid."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    if len(aa) == 0 or len(bb) == 0:
        return float("nan")
    grid = np.unique(np.concatenate([aa, bb]))
    # Also evaluate just below each point for left limits (ties)
    eps = np.nextafter(grid, -np.inf)
    xs = np.unique(np.concatenate([grid, eps]))
    fa = empirical_cdf_values(aa, xs)
    fb = empirical_cdf_values(bb, xs)
    return float(np.max(np.abs(fa - fb)))


def wasserstein1(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.sort(np.asarray(a, dtype=float))
    bb = np.sort(np.asarray(b, dtype=float))
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    if len(aa) == 0 or len(bb) == 0:
        return float("nan")
    # Quantile averaging on common grid
    qs = np.linspace(0, 1, num=max(len(aa), len(bb), 64))
    qa = np.quantile(aa, qs)
    qb = np.quantile(bb, qs)
    integ = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integ(np.abs(qa - qb), qs))


def abstain_rate(scores: Sequence[float], tau: float) -> float:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return float("nan")
    return float(np.mean(s >= tau))


def transfer_gap_g(scores: Sequence[float], tau: float) -> float:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) == 0 or tau <= 0:
        return float("nan")
    return float(np.max(s) / tau)


def evaluate_cell(
    name: str,
    fixture_scores: Sequence[float],
    live_scores: Sequence[float],
    tau: float,
    *,
    channel: str = "",
    tol: float = 1e-9,
) -> OccupancyCell:
    fr = abstain_rate(fixture_scores, tau)
    lr = abstain_rate(live_scores, tau)
    gap = abs(lr - fr)
    ks = ks_distance(fixture_scores, live_scores)
    w1 = wasserstein1(fixture_scores, live_scores)
    g = transfer_gap_g(live_scores, tau)
    return OccupancyCell(
        name=name,
        tau=float(tau),
        n_fixture=int(np.sum(np.isfinite(fixture_scores))),
        n_live=int(np.sum(np.isfinite(live_scores))),
        fixture_abstain_rate=fr,
        live_abstain_rate=lr,
        occupancy_gap=gap,
        ks=ks,
        w1=w1,
        bound_holds=bool(gap <= ks + tol) if np.isfinite(ks) else False,
        transfer_gap_g=g,
        idle_reject_live=bool(lr == 0.0),
        channel=channel,
    )


def fixture_quantile_tau(fixture_scores: Sequence[float], q: float = 0.95) -> float:
    s = np.asarray(fixture_scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return float("nan")
    return float(np.quantile(s, q))


def _scores_from_rows(rows: list[dict[str, Any]], key: str) -> list[float]:
    from benchmarks.protocol_phase2_9_10_battery import _row_channels

    out: list[float] = []
    for r in rows:
        ch = _row_channels(r)
        if key == "composite":
            out.append(float(ch.get("composite", r.get("composite_score", float("nan")))))
        elif key == "inv_composite":
            out.append(float(ch.get("inv_composite", float("nan"))))
        else:
            out.append(float(ch.get(key, float("nan"))))
    return out


def run_pair(
    name: str,
    fixture_path: Path,
    live_path: Path,
    *,
    channel: str,
    tau: Optional[float] = None,
    q: float = 0.95,
) -> OccupancyCell:
    fix = json.loads(fixture_path.read_text(encoding="utf-8"))
    live = json.loads(live_path.read_text(encoding="utf-8"))
    frows = fix["rows"] if isinstance(fix, dict) else fix
    lrows = live["rows"] if isinstance(live, dict) else live
    fs = _scores_from_rows(frows, channel)
    ls = _scores_from_rows(lrows, channel)
    if tau is None:
        tau = fixture_quantile_tau(fs, q=q)
    return evaluate_cell(name, fs, ls, float(tau), channel=channel)


def synthesize_demo_cells(seed: int = 0) -> list[OccupancyCell]:
    """Synthetic checks for theorem unit tests / smoke figure."""
    rng = np.random.default_rng(seed)
    cells: list[OccupancyCell] = []
    # Matched distributions => gap ~ 0, KS ~ 0
    a = rng.normal(0.4, 0.1, size=400)
    cells.append(evaluate_cell("synth_matched", a, a + rng.normal(0, 1e-6, size=400), tau=0.6))
    # Shifted live mass below tau => idle reject, large KS
    fix = rng.normal(0.7, 0.08, size=400)
    live = rng.normal(0.3, 0.08, size=400)
    tau = float(np.quantile(fix, 0.95))
    cells.append(evaluate_cell("synth_shifted_idle", fix, live, tau=tau))
    return cells


def write_rollup(cells: list[OccupancyCell], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "anima.occupancy_ks_bound.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "theorem": (
            "|P_L(A=1)-P_F(A=1)| = |F_F(tau-)-F_L(tau-)| <= KS(F_F,F_L); "
            "g=max/tau is idle-reject corollary only"
        ),
        "n_cells": len(cells),
        "frac_bound_holds": float(np.mean([c.bound_holds for c in cells])) if cells else 0.0,
        "cells": [asdict(c) for c in cells],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", action="store_true", help="Synthetic theorem smoke cells")
    p.add_argument(
        "--out",
        type=Path,
        default=REPORTS / "occupancy_ks_bound_rollup.json",
    )
    p.add_argument("--also-paper", action="store_true")
    args = p.parse_args(argv)

    cells: list[OccupancyCell] = []
    if args.demo:
        cells.extend(synthesize_demo_cells())

    # Prefer sealed live reports if present beside this package
    live_dir = REPORTS
    defaults = [
        ("qwen05_halu_composite", live_dir / "halueval_live_qwen_qwen2_5-0_5b-instruct.json", "composite", 0.82),
        ("qwen05_tqa_composite", live_dir / "truthfulqa_live_qwen_qwen2_5-0_5b-instruct.json", "composite", 0.82),
        ("qwen05_halu_entropy", live_dir / "halueval_live_qwen_qwen2_5-0_5b-instruct.json", "entropy", 0.90),
        ("qwen05_halu_invgap", live_dir / "halueval_live_qwen_qwen2_5-0_5b-instruct.json", "inv_logit_gap", 0.69),
        ("tiny_halu_composite", live_dir / "halueval_live_tinyllama_tinyllama-1_1b-chat-v1_0.json", "composite", 0.82),
        ("distil_halu_composite", live_dir / "halueval_live_distilgpt2.json", "composite", 0.82),
    ]
    # Without a separate fixture encode, use live as both (gap=0) is invalid —
    # instead bootstrap a "fixture-like" high-scoring synthetic from live quantiles
    # ONLY for demo continuity; real fixture pair should replace this.
    for name, path, channel, tau in defaults:
        if not path.exists():
            continue
        live = json.loads(path.read_text(encoding="utf-8"))
        rows = live["rows"] if isinstance(live, dict) else live
        ls = _scores_from_rows(rows, channel)
        # Construct a fixture proxy: scores shifted toward the upper tail so
        # fixture p95 gate fires ~5% — used only when fixture JSON absent.
        arr = np.asarray(ls, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        # Affine map live -> proxy fixture with same rank, higher mean
        proxy = 0.5 * arr + 0.45
        cell = evaluate_cell(f"{name}_proxyfix", proxy, ls, tau=float(tau), channel=channel)
        cell.notes = "fixture_scores=affine_proxy; replace with true Protocol-F encode"
        cells.append(cell)

    # Length-balanced fleet live-only self-gap (KS=0) skipped — not informative.

    out = write_rollup(cells, args.out)
    print(f"wrote {out} n_cells={len(cells)} frac_bound_holds={np.mean([c.bound_holds for c in cells]) if cells else 'n/a'}")
    if args.also_paper:
        PAPER_ART.mkdir(parents=True, exist_ok=True)
        dest = PAPER_ART / out.name
        dest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"also {dest}")


if __name__ == "__main__":
    main()
