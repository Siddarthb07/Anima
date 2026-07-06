"""Tests for rolling readout stability and guard gate."""

from types import SimpleNamespace

from core.stability import analyze_readout_stability, apply_guard_gate, merge_stability_flags


def _ro(valence: float, abstain: bool = False):
    return SimpleNamespace(
        affect={"valence": valence, "arousal": 0.5, "uncertainty": 0.3},
        guard=SimpleNamespace(abstain_recommended=abstain),
        flags={},
        region="neutral",
        region_analog="neutral",
    )


def test_stable_readouts_score_high():
    readouts = [_ro(0.1), _ro(0.12), _ro(0.11), _ro(0.09)]
    per_token, summary = analyze_readout_stability(readouts, window=4)
    assert summary["stability_score"] > 0.7
    assert summary["unstable_token_count"] == 0
    assert len(per_token) == 4


def test_volatile_readouts_flag_unstable():
    readouts = [_ro(0.8), _ro(-0.7), _ro(0.6), _ro(-0.5), _ro(0.4)]
    per_token, summary = analyze_readout_stability(readouts, window=4)
    assert summary["unstable_token_count"] >= 1
    assert summary["max_valence_swing"] > 0.3


def test_guard_gate_relabels_unstable_tokens():
    readouts = [_ro(0.8), _ro(-0.7), _ro(0.6), _ro(-0.5)]
    per_token, _ = analyze_readout_stability(readouts, window=4)
    merge_stability_flags(readouts, per_token)
    gated = apply_guard_gate(readouts, per_token, guard_mode="gate")
    assert gated >= 1
    assert any(r.flags.get("gated") for r in readouts)
    assert any(r.region == "high-uncertainty" for r in readouts)


def test_guard_gate_observe_noop():
    readouts = [_ro(0.8), _ro(-0.7)]
    per_token, _ = analyze_readout_stability(readouts, window=2)
    gated = apply_guard_gate(readouts, per_token, guard_mode="observe")
    assert gated == 0
