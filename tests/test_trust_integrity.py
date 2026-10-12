"""Tests for Trust Integrity Gate (green-shield claim lock)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.trust_integrity import (
    SuiteIntegrity,
    enforce_trust_language,
    evaluate_trust_integrity,
    transfer_gap,
    channel_sign_conflict,
)

REPORTS = Path(__file__).resolve().parent.parent / "benchmarks" / "reports"
QWEN_SLUG = "qwen_qwen2_5-0_5b-instruct"


def test_transfer_gap_definition():
    assert transfer_gap(1.0, 0.2469) == pytest.approx(0.7531, abs=1e-4)


def test_channel_sign_conflict():
    assert channel_sign_conflict(0.52, 0.25) is True
    assert channel_sign_conflict(0.52, 0.55) is False
    assert channel_sign_conflict(None, 0.5) is None


def test_enforce_locks_green_tier():
    locked = enforce_trust_language(
        claim_allowed=False, proposed_tier="HIGH", abstain_recommended=False
    )
    assert locked["trust_ui_claim_allowed"] is False
    assert locked["tier"] == "LOW"
    assert locked["abstain_recommended"] is True
    assert locked["badge"] == "locked"

    open_ = enforce_trust_language(
        claim_allowed=True, proposed_tier="HIGH", abstain_recommended=False
    )
    assert open_["tier"] == "HIGH"
    assert open_["badge"] == "allowed"


def test_qwen_receipt_claim_denied():
    """Locked Qwen numbers must FAIL the gate (that is the paper-grade finding)."""
    suites = {
        "halueval": SuiteIntegrity(
            suite="halueval",
            protocol_f_auroc=1.0,
            protocol_l_auroc=0.2469,
            transfer_gap=transfer_gap(1.0, 0.2469),
            auroc_ci95=None,
            never_abstains=True,
            n_live=80,
            polarity_verdict="anti_aligned_ranking",
            probe_only_auroc=0.5228,
            fused_only_auroc=0.2487,
            channel_sign_conflict=True,
        ),
        "truthfulqa": SuiteIntegrity(
            suite="truthfulqa",
            protocol_f_auroc=1.0,
            protocol_l_auroc=0.4587,
            transfer_gap=transfer_gap(1.0, 0.4587),
            auroc_ci95=None,
            never_abstains=True,
            n_live=80,
            polarity_verdict="anti_aligned_ranking",
            probe_only_auroc=0.6569,
            fused_only_auroc=0.4391,
            channel_sign_conflict=True,
        ),
    }
    r = evaluate_trust_integrity(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        model_slug=QWEN_SLUG,
        suites=suites,
    )
    assert r.trust_ui_claim_allowed is False
    assert r.lab_verified is True
    assert r.seal_sha256
    assert any("anti_aligned" in g for g in r.failed_gates)
    assert any("transfer_gap" in g for g in r.failed_gates)
    assert any("never_abstains" in g for g in r.failed_gates)


@pytest.mark.skipif(
    not (REPORTS / f"live_guard_{QWEN_SLUG}.json").is_file(),
    reason="live guard JSON missing",
)
def test_runner_writes_qwen_receipt():
    from benchmarks.run_trust_integrity_gate import run_for_slug
    from core.trust_integrity import write_receipt

    receipt = run_for_slug(
        slug=QWEN_SLUG,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reports=REPORTS,
    )
    assert receipt.trust_ui_claim_allowed is False
    out = REPORTS / f"trust_integrity_{QWEN_SLUG}.json"
    write_receipt(receipt, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == "anima.trust_integrity.v1"
    assert data["trust_ui_claim_allowed"] is False
