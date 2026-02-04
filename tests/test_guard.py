from core.guard import evaluate_guard


def test_guard_abstain_on_high_fused():
    g = evaluate_guard(
        affect={"valence": 0.0, "arousal": 0.5, "uncertainty": 0.9},
        uncertainty_signals={"fused": 0.9, "entropy": 0.9, "logit_gap": 0.9, "attn_entropy": 0.5},
        token_text="perhaps maybe unclear",
    )
    assert g.abstain_recommended or "high_fused_uncertainty" in g.reasons
