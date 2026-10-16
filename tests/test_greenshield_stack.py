"""Tests for GreenShieldBench, misleading-green, polarity recovery, Trust Badge Console."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.misleading_green import (
    default_fixture_policy,
    misleading_green_summary,
    polarity_aware_policy,
)
from benchmarks.trust_badge_console import load_console_state, write_console
from core.trust_integrity import transfer_gap

REPORTS = Path(__file__).resolve().parent.parent / "benchmarks" / "reports"
QWEN = "qwen_qwen2_5-0_5b-instruct"
TINY = "tinyllama_tinyllama-1_1b-chat-v1_0"


def _halueval_rows(slug: str) -> list:
    path = REPORTS / f"halueval_live_{slug}.json"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def test_misleading_green_construct_on_qwen_halueval():
    rows = _halueval_rows(QWEN)
    mg = misleading_green_summary(
        rows,
        label_key="is_hallucination",
        policy_name="fixture_thresholds",
        decide=default_fixture_policy,
    )
    assert mg["construct"] == "simulated_badge_replay_not_user_study"
    # Qwen never abstains on locked live → every hall row is misleading-green
    assert mg["n_abstain"] == 0
    assert mg["misleading_green_rate"] > 0.3
    assert "user study" not in mg["honesty_note"].lower() or "not" in mg["honesty_note"].lower()


def test_polarity_aware_policy_invert_is_named_not_silent():
    decide = polarity_aware_policy(invert_composite=True, disable_fused=False)
    rows = _halueval_rows(QWEN)[:3]
    # Must run without mutating production config; returns bools only
    outs = [decide(r) for r in rows]
    assert all(isinstance(x, bool) for x in outs)


def test_greenshield_bench_qwen_baseline_fails_honestly():
    from benchmarks.greenshield_bench import score_model

    score = score_model(QWEN, "Qwen/Qwen2.5-0.5B-Instruct", REPORTS, adapted=False)
    assert score["passed"] is False
    assert "tig_claim_allowed" in score["failed_checks"]
    assert score["honesty"]["not_hallucination_sota"] is True
    he = score["suites"]["halueval"]
    assert he["protocol_l_auroc"] == pytest.approx(0.2469, abs=1e-3)
    assert he["transfer_gap"] == pytest.approx(transfer_gap(1.0, 0.2469), abs=1e-3)


@pytest.mark.skipif(
    not (REPORTS / f"protocol_l_prime_{QWEN}.json").is_file(),
    reason="Protocol L′ report missing",
)
def test_greenshield_bench_adapted_qwen_passes():
    from benchmarks.greenshield_bench import score_model

    score = score_model(QWEN, "Qwen/Qwen2.5-0.5B-Instruct", REPORTS, adapted=True)
    assert score["passed"] is True
    assert score["failed_checks"] == []
    assert score["trust_integrity"]["trust_ui_claim_allowed"] is True
    assert score["suites"]["halueval"]["protocol_l_auroc"] >= 0.55


@pytest.mark.skipif(
    not (REPORTS / f"protocol_l_prime_{TINY}.json").is_file(),
    reason="Protocol L′ report missing",
)
def test_greenshield_bench_adapted_tinyllama_passes():
    from benchmarks.greenshield_bench import score_model

    score = score_model(TINY, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", REPORTS, adapted=True)
    assert score["passed"] is True
    assert score["trust_integrity"]["trust_ui_claim_allowed"] is True
    assert score["suites"]["halueval"]["protocol_l_auroc"] >= 0.55


def test_polarity_aware_recovery_held_out_only():
    from benchmarks.polarity_aware_recovery import run_for_slug

    report = run_for_slug(QWEN, REPORTS, seed=42)
    assert report["suites"]["halueval"]["status"] == "ok"
    assert report["suites"]["halueval"]["honesty"]["silent_label_flip"] is False
    assert report["suites"]["halueval"]["honesty"]["production_guard_config_unchanged"] is True
    d = report["suites"]["halueval"]["diagnose_fold"]
    assert d["anti_aligned"] is True
    assert d["chosen_policy"] == "invert_composite_abstain"
    # Held-out AUROC after invert should beat baseline anti-aligned (~0.25)
    prim = report["suites"]["halueval"]["primary_held_out"]
    assert prim["auroc_composite"] > 0.5


def test_trust_badge_console_preserves_baseline_collapse():
    st = load_console_state(QWEN, REPORTS)
    assert st["honesty"]["fixture_auroc_cannot_unlock"] is True
    assert st["honesty"]["baseline_collapse_preserved"] is True
    # Baseline must stay denied even if adapted unlocks
    assert st.get("baseline_claim_allowed") is False


def test_trust_badge_console_writes_html(tmp_path: Path):
    out = write_console(REPORTS, [QWEN, TINY])
    assert Path(out["json"]).is_file()
    assert Path(out["html"]).is_file()
    html = Path(out["html"]).read_text(encoding="utf-8")
    assert "fixture AUROC" in html.lower() or "cannot unlock" in html.lower()
