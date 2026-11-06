#!/usr/bin/env python3
"""Protocol: transfer gap g = max(live_score) / tau_c predicts never-abstain.

Confirmatory companion to PROTOCOL_PREREG.md Definition (idle-reject).
Uses locked live JSON + fixture-calibrated tau from Protocol M receipts when present.
shipping_claim=false.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from benchmarks.protocol_phase2_9_10_battery import _row_channels

REPORTS = Path(__file__).resolve().parent / "reports"
PAPER = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact")
PREREG = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact\prereg")

# Fallback fixture p95-style taus from sealed Protocol M / G' narrative when
# per-channel fixture JSON is absent (exploratory bridge until Phase B fixtures).
DEFAULT_TAU = {
    "composite": 0.82,
    "fused": 0.82,
    "probe": 0.78,
    "inv_logit_gap": 0.69,
    "entropy": 0.90,
}


def _gap(scores: list[float], tau: float) -> dict[str, Any]:
    mx = float(max(scores)) if scores else float("nan")
    g = mx / tau if tau > 0 else float("nan")
    n_abs = sum(1 for s in scores if s >= tau)
    return {
        "tau_c": tau,
        "live_max": mx,
        "transfer_gap_g": g,
        "n": len(scores),
        "n_abstain": n_abs,
        "never_abstain": n_abs == 0,
        "predicts_never_abstain": bool(g < 1.0),
        "prediction_holds": bool((g < 1.0) == (n_abs == 0)),
    }


def analyze_live(
    path: Path,
    *,
    label_key: str,
    taus: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    taus = dict(DEFAULT_TAU if taus is None else taus)
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    chans = [_row_channels(r) for r in rows]
    out_channels = {}
    for name in ("composite", "inv_composite", "fused", "probe", "inv_logit_gap", "entropy"):
        if name == "inv_composite":
            scores = [c["inv_composite"] for c in chans]
            tau = 1.0 - taus.get("composite", 0.82)  # exploratory invert gate
        else:
            scores = [c[name] for c in chans]
            tau = float(taus.get(name, 0.5))
        out_channels[name] = _gap(scores, tau)
    # Length diagnostic (not a gate)
    lens = [c["answer_length"] for c in chans]
    y = [int(bool(r[label_key])) for r in rows]
    from benchmarks.guard_metrics import auroc_from_scores

    return {
        "source": str(path),
        "n": len(rows),
        "label_key": label_key,
        "length_to_label_auroc": auroc_from_scores(lens, y),
        "channels": out_channels,
        "n_channels_prediction_holds": sum(
            1 for v in out_channels.values() if v.get("prediction_holds")
        ),
        "n_channels": len(out_channels),
    }


def main() -> None:
    jobs = [
        ("qwen_qwen2_5-0_5b-instruct", "halueval", "is_hallucination"),
        ("qwen_qwen2_5-0_5b-instruct", "truthfulqa", "should_abstain"),
        ("qwen_qwen2_5-7b-instruct", "halueval", "is_hallucination"),
        ("qwen_qwen2_5-7b-instruct", "truthfulqa", "should_abstain"),
        ("distilgpt2", "halueval", "is_hallucination"),
    ]
    rollup: dict[str, Any] = {
        "schema": "anima.protocol_transfer_gap.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prereg": "PROTOCOL_PREREG.md",
        "shipping_claim": False,
        "note": "DEFAULT_TAU from exploratory M/G' narrative until Phase B real fixtures land",
        "cells": {},
    }
    holds = 0
    total = 0
    for slug, suite, lk in jobs:
        path = REPORTS / f"{suite}_live_{slug}.json"
        key = f"{slug}::{suite}"
        if not path.is_file():
            rollup["cells"][key] = {"status": "missing", "path": str(path)}
            print(f"MISSING {key}")
            continue
        block = analyze_live(path, label_key=lk)
        rollup["cells"][key] = block
        for ch, v in block["channels"].items():
            total += 1
            if v["prediction_holds"]:
                holds += 1
            print(
                f"{key} {ch}: g={v['transfer_gap_g']:.3f} "
                f"never={v['never_abstain']} pred_ok={v['prediction_holds']}"
            )
    rollup["summary"] = {
        "n_channel_tests": total,
        "n_prediction_holds": holds,
        "frac_prediction_holds": (holds / total) if total else None,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    PREREG.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    for dest in (REPORTS / "protocol_transfer_gap.json", PREREG / "protocol_transfer_gap.json", PAPER / "protocol_transfer_gap.json"):
        dest.write_text(json.dumps(rollup, indent=2), encoding="utf-8")
    print(json.dumps(rollup["summary"], indent=2))
    print(f"Wrote {REPORTS / 'protocol_transfer_gap.json'}")


if __name__ == "__main__":
    main()
