import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HF_TESTS") != "1",
    reason="Set RUN_HF_TESTS=1 to load HF weights for quant smoke test.",
)


def test_dynamic_int8_distilgpt2_extract_smoke(monkeypatch):
    """Hooks + one-token extract work with ANIMA_LOAD_DYNAMIC_INT8 on CPU."""
    monkeypatch.setenv("ANIMA_LOAD_DYNAMIC_INT8", "1")
    monkeypatch.setenv("ANIMA_FORCE_CPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    from core.extractor import ActivationExtractor
    from probes.linear_probe import AffectProbe

    ex = ActivationExtractor("distilgpt2", device="cpu")
    probe = AffectProbe(ex.hidden_dim, len(ex.layer_indices))
    rows = ex.extract("Hello", max_new_tokens=2)
    assert len(rows) >= 1
    affect = probe.predict(rows[0]["activations"])
    assert "valence" in affect
    ex.cleanup()
