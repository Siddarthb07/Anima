"""Unit tests for chat-template wrapping, guard ablations, and narratives validator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from core.guard import evaluate_guard
from core.layer_config import LAYER_CONFIG
from core.prompt_format import format_user_text, uses_chat_template


def _load_validate_root():
    path = Path(__file__).resolve().parent.parent / "scripts" / "validate_narratives_root.py"
    spec = importlib.util.spec_from_file_location("validate_narratives_root", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.validate_root


def test_instruct_models_flag_chat_template():
    assert uses_chat_template("Qwen/Qwen2.5-0.5B-Instruct") is True
    assert uses_chat_template("TinyLlama/TinyLlama-1.1B-Chat-v1.0") is True
    assert uses_chat_template("distilgpt2") is False
    assert LAYER_CONFIG["distilgpt2"].get("use_chat_template") in (None, False)


def test_format_user_text_passthrough_without_template():
    class Tok:
        pass

    assert format_user_text(Tok(), "hello", enable=False) == "hello"


def test_guard_ablation_disables_fused():
    affect = {"uncertainty": 0.2}
    sigs = {"fused": 0.95, "entropy": 0.9, "logit_gap": 0.9, "attn_entropy": 0.9}
    full = evaluate_guard(affect=affect, uncertainty_signals=sigs, token_text="yes")
    assert full.abstain_recommended is True
    ablated = evaluate_guard(
        affect=affect,
        uncertainty_signals=sigs,
        token_text="yes",
        disabled_signals=("fused",),
    )
    assert ablated.abstain_recommended is False
    assert "high_fused_uncertainty" not in ablated.reasons


def test_guard_ablation_hedging_only():
    affect = {"uncertainty": 0.1}
    sigs = {"fused": 0.1}
    text = "maybe perhaps possibly unclear"
    g = evaluate_guard(
        affect=affect,
        uncertainty_signals=sigs,
        token_text=text,
        disabled_signals=("fused", "probe_uncertainty"),
    )
    assert g.abstain_recommended is False or "lexical_hedging" in g.reasons


def test_validate_synthetic_narratives_layout():
    validate_root = _load_validate_root()
    root = Path(__file__).resolve().parent.parent / "data" / "narratives_minimal"
    report = validate_root(root)
    assert report["layout_ok"] is True
    assert report["recommended_probe_origin"] == "narratives_fMRI_synthetic_minimal"
