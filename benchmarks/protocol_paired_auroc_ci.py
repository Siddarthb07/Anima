#!/usr/bin/env python3
"""Paired bootstrap CIs on AUROC differences (Finding 8 honesty).

Uses sealed hall_probe heldout_scores + live-channel scores on same fold.
CPU-only. shipping_claim=false.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.guard_metrics import auroc_from_scores
from benchmarks.protocol_phase2_9_10_battery import _row_channels
from scripts.protocol_r_retune import question_holdout_split

REPORTS = Path(__file__).resolve().parent / "reports"
PAPER = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact")
PREREG = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact\prereg")


def _auroc_diff_boot(
    scores_a: list[float],
    scores_b: list[float],
    labels: list[int],
    *,
    B: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    point = auroc_from_scores(scores_a, labels) - auroc_from_scores(scores_b, labels)
    diffs = []
    ya = np.asarray(scores_a, float)
    yb = np.asarray(scores_b, float)
    lab = np.asarray(labels, int)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        diffs.append(
            auroc_from_scores(ya[idx].tolist(), lab[idx].tolist())
            - auroc_from_scores(yb[idx].tolist(), lab[idx].tolist())
        )
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "delta_auroc": point,
        "ci95": [float(lo), float(hi)],
        "covers_zero": bool(lo <= 0 <= hi),
        "B": B,
        "seed": seed,
    }


def main() -> None:
    slug = "qwen_qwen2_5-0_5b-instruct"
    live = json.loads((REPORTS / f"truthfulqa_live_{slug}.json").read_text(encoding="utf-8"))["rows"]
    hall_path = REPORTS / "hall_probe_qwen_qwen2_5_0_5b_instruct_truthfulqa.json"
    if not hall_path.is_file():
        hall_path = PAPER / "hall_probe_qwen_qwen2_5_0_5b_instruct_truthfulqa.json"
    hall = json.loads(hall_path.read_text(encoding="utf-8"))
    qs = [str(r.get("question") or "") for r in live]
    _, test_m = question_holdout_split(qs, seed=42)
    test = [r for r, m in zip(live, test_m) if m]
    y = [int(bool(r["should_abstain"])) for r in test]
    ch = [_row_channels(r) for r in test]
    auroc_comp = auroc_from_scores([c["inv_composite"] for c in ch], y)
    # G' approx
    gprime = [0.65 * c["inv_logit_gap"] + 0.35 * c["probe"] for c in ch]
    hall_s = [float(x) for x in hall["heldout_scores"]]
    assert len(hall_s) == len(y), (len(hall_s), len(y))
    out = {
        "schema": "anima.paired_auroc_ci.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": "truthfulqa",
        "model_slug": slug,
        "n_heldout": len(y),
        "pilot": True,
        "auroc": {
            "hall_probe": auroc_from_scores(hall_s, y),
            "sys_a_inv_composite": auroc_comp,
            "gprime": auroc_from_scores(gprime, y),
            "channel_probe": auroc_from_scores([c["probe"] for c in ch], y),
        },
        "paired_delta_vs_hall": {
            "hall_minus_sys_a": _auroc_diff_boot(hall_s, [c["inv_composite"] for c in ch], y),
            "hall_minus_gprime": _auroc_diff_boot(hall_s, gprime, y),
            "hall_minus_probe": _auroc_diff_boot(hall_s, [c["probe"] for c in ch], y),
        },
        "shipping_claim": False,
        "note": "If CI covers 0, do not claim significant ranking win",
    }
    for dest_dir in (REPORTS, PAPER, PREREG):
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "protocol_paired_auroc_ci.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8"
        )
    print(json.dumps(out["auroc"], indent=2))
    print(json.dumps(out["paired_delta_vs_hall"], indent=2))


if __name__ == "__main__":
    main()
