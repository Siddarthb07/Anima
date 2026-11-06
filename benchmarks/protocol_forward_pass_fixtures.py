#!/usr/bin/env python3
"""Phase B: build fixture τ_c from real forward-pass claim-encode rows.

Splits locked live JSON by question into fixture_cal / live_transfer folds.
Calibrates per-channel p95 on fixture_cal; evaluates transfer gap on live_transfer.
Pre-baked fixture remains smoke-only (not used here).

Requires existing *_live_*.json (CPU OK once rows exist). GPU needed to *create* rows.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.protocol_phase2_9_10_battery import _row_channels
from benchmarks.protocol_transfer_gap import _gap
from scripts.protocol_r_retune import question_holdout_split

REPORTS = Path(__file__).resolve().parent / "reports"
PREREG = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact\prereg")
CHANNELS = ("composite", "fused", "probe", "inv_logit_gap", "entropy")


def calibrate_p95(values: list[float], q: float = 0.95) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(values, q))


def run(slug: str, suite: str, label_key: str, seed: int = 42) -> dict[str, Any]:
    path = REPORTS / f"{suite}_live_{slug}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    qs = [str(r.get("question") or "") for r in rows]
    fix_m, live_m = question_holdout_split(qs, seed=seed)
    fix = [r for r, m in zip(rows, fix_m) if m]
    live = [r for r, m in zip(rows, live_m) if m]
    fix_ch = [_row_channels(r) for r in fix]
    live_ch = [_row_channels(r) for r in live]
    taus = {c: calibrate_p95([ch[c] for ch in fix_ch]) for c in CHANNELS}
    gaps = {c: _gap([ch[c] for ch in live_ch], taus[c]) for c in CHANNELS}
    return {
        "schema": "anima.forward_pass_fixture.v1",
        "model_slug": slug,
        "suite": suite,
        "seed": seed,
        "n_fixture_cal": len(fix),
        "n_live_transfer": len(live),
        "tau_c_p95_from_forward_pass": taus,
        "transfer_gaps_on_live": gaps,
        "prebaked_fixture_used": False,
        "shipping_claim": False,
        "source": str(path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    rollup = {
        "schema": "anima.forward_pass_fixture.rollup.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shipping_claim": False,
        "models": {},
    }
    for slug, suite, lk in (
        ("qwen_qwen2_5-0_5b-instruct", "halueval", "is_hallucination"),
        ("qwen_qwen2_5-0_5b-instruct", "truthfulqa", "should_abstain"),
        ("qwen_qwen2_5-7b-instruct", "halueval", "is_hallucination"),
        ("distilgpt2", "halueval", "is_hallucination"),
    ):
        key = f"{slug}::{suite}"
        path = REPORTS / f"{suite}_live_{slug}.json"
        if not path.is_file():
            rollup["models"][key] = {"status": "missing"}
            print("MISSING", key)
            continue
        block = run(slug, suite, lk)
        rollup["models"][key] = block
        print(key, "taus", {k: round(v, 3) for k, v in block["tau_c_p95_from_forward_pass"].items()})
        for c, g in block["transfer_gaps_on_live"].items():
            print(f"  {c}: g={g['transfer_gap_g']:.3f} never={g['never_abstain']} ok={g['prediction_holds']}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    PREREG.mkdir(parents=True, exist_ok=True)
    for dest in (
        REPORTS / "protocol_forward_pass_fixtures.json",
        PREREG / "protocol_forward_pass_fixtures.json",
    ):
        dest.write_text(json.dumps(rollup, indent=2), encoding="utf-8")
    print("Wrote", REPORTS / "protocol_forward_pass_fixtures.json")


if __name__ == "__main__":
    main()
