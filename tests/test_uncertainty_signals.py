"""
Uncertainty metric sanity checks (no Hugging Face download).

Uses bare ActivationExtractor instances via object.__new__ to call math helpers.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.extractor import ActivationExtractor


def _make_extractor_shell():
    return object.__new__(ActivationExtractor)


def test_normalized_entropy_near_one_for_uniform_logits():
    ext = _make_extractor_shell()
    vocab = 1024
    logits = torch.zeros(vocab)
    e = ActivationExtractor._compute_entropy(ext, logits)
    assert e > 0.995


def test_normalized_entropy_near_zero_for_peaked_logits():
    ext = _make_extractor_shell()
    vocab = 1024
    logits = torch.full((vocab,), -50.0)
    logits[42] = 50.0
    e = ActivationExtractor._compute_entropy(ext, logits)
    assert e < 0.02


def test_logit_gap_uncertainty_high_when_top_two_close():
    ext = _make_extractor_shell()
    logits = torch.tensor([0.0, 0.01, -10.0, -10.0])
    g = ActivationExtractor._compute_logit_gap(ext, logits)
    assert g > 0.9


def test_logit_gap_uncertainty_low_when_top_two_far():
    ext = _make_extractor_shell()
    logits = torch.tensor([100.0, 0.0, -50.0])
    g = ActivationExtractor._compute_logit_gap(ext, logits)
    assert g < 0.05


def test_fused_uncertainty_blends_components():
    ext = _make_extractor_shell()
    fused_high = ActivationExtractor._fuse_uncertainty(ext, 1.0, 1.0, 1.0)
    # Avoid exact 0.5 blend edge (0.35*1 + 0.35*0 + 0.30*0.5 == 0.50).
    fused_mixed = ActivationExtractor._fuse_uncertainty(ext, 1.0, 0.2, 0.5)
    assert fused_high == 1.0
    assert 0.5 < fused_mixed < 1.0


def test_attention_entropy_high_when_attention_is_uniform():
    ext = _make_extractor_shell()
    seq = 16
    heads = 4
    attn = torch.full((1, heads, seq, seq), 1.0 / seq, dtype=torch.float32)
    layer = (attn,)
    e = ActivationExtractor._compute_attention_entropy(ext, layer)
    assert e > 0.95


def test_attention_entropy_low_when_attention_is_peaked():
    ext = _make_extractor_shell()
    seq = 8
    heads = 2
    attn = torch.zeros((1, heads, seq, seq), dtype=torch.float32)
    attn[..., -1] = 1.0
    layer = (attn,)
    e = ActivationExtractor._compute_attention_entropy(ext, layer)
    assert e < 0.05


def test_last_none_attention_layer_skipped():
    """Matches DistilGPT-style stacks where tail layers omit attentions."""
    ext = _make_extractor_shell()
    seq = 6
    heads = 2
    good = torch.full((1, heads, seq, seq), 1.0 / seq, dtype=torch.float32)
    bad = None
    e = ActivationExtractor._compute_attention_entropy(ext, (bad, bad, good))
    assert e > 0.9
