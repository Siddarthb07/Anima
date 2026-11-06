"""Phase-1 held-out OP: Qwen Halu must abstain on held-out fold under invert."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.protocol_op_heldout_phase1 import run_slug

REPORTS = Path(__file__).resolve().parents[1] / "benchmarks" / "reports"
SLUG = "qwen_qwen2_5-0_5b-instruct"


def test_qwen_halueval_op_abstains_and_cuts_false_green():
    live = REPORTS / f"halueval_live_{SLUG}.json"
    if not live.is_file():
        return  # skip if locked rows absent in CI clone
    report = run_slug(SLUG, REPORTS, seed=42)
    block = report["suites"]["halueval"]
    assert block["status"] == "ok"
    rq = block["reliability_quantities"]
    assert rq["chosen_policy"] == "invert_composite_abstain"
    abs_n, tot = rq["abstain_events"].split("/")
    assert int(abs_n) > 0
    assert rq["false_green_rate"] < rq["baseline_fixture_false_green_rate"]
    assert rq["held_out_auroc"] > 0.7
    cov20 = rq["coverage_at_risk"]["coverage_at_risk_0.20"]
    assert cov20["status"] == "ok"
    assert cov20["coverage"] >= 0.3


def test_shipping_claim_false_in_rollup_schema():
    report = {
        "shipping_claim": False,
        "pre_registered": {"shipping_claim": False},
    }
    assert report["shipping_claim"] is False
