"""Tests for API request limits and public demo policy."""

import pytest

from core.limits import (
    PUBLIC_DEMO_MODELS,
    assert_model_allowed,
    clamp_max_new_tokens,
    public_mode_enabled,
)


def test_clamp_max_new_tokens():
    assert clamp_max_new_tokens(1) == 1
    assert clamp_max_new_tokens(99999) == 512
    assert clamp_max_new_tokens(200) == 200


def test_public_mode_blocks_large_models(monkeypatch):
    monkeypatch.setenv("ANIMA_PUBLIC_MODE", "1")
    assert public_mode_enabled()
    assert_model_allowed("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    with pytest.raises(ValueError, match="not enabled on public demo"):
        assert_model_allowed("distilgpt2")


def test_public_mode_off_allows_all(monkeypatch):
    monkeypatch.delenv("ANIMA_PUBLIC_MODE", raising=False)
    assert not public_mode_enabled()
    assert_model_allowed("distilgpt2")


def test_public_demo_model_set():
    assert "hf-internal-testing/tiny-random-gpt2" in PUBLIC_DEMO_MODELS
    assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0" in PUBLIC_DEMO_MODELS
