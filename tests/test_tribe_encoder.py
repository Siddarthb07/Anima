import numpy as np
import pytest

from alignment.tribe_encoder import ROI_DEFINITIONS, TRIBEv2Encoder, tribe_seed


def test_tribe_seed_stable():
    assert tribe_seed("distilgpt2") == tribe_seed("distilgpt2")
    assert tribe_seed("a") != tribe_seed("b")


def test_encode_layer_activations_mean_layers():
    enc = TRIBEv2Encoder(16, seed=12345)
    h0 = np.random.randn(16).astype(np.float64)
    h1 = np.random.randn(16).astype(np.float64)
    a = enc.encode_layer_activations({0: h0, 1: h1})
    b = enc.encode_layer_activations({0: h0})
    assert set(a.keys()) == set(ROI_DEFINITIONS.keys())
    assert all(isinstance(v, float) for v in a.values())
    assert a != b


def test_derived_va_bounds():
    enc = TRIBEv2Encoder(8, seed=1)
    fake = {roi: float(np.sin(i)) for i, roi in enumerate(ROI_DEFINITIONS)}
    va = enc.derived_va_from_rois(fake)
    assert -1 <= va["valence"] <= 1
    assert 0 <= va["arousal"] <= 1


def test_hidden_dim_mismatch_errors():
    enc = TRIBEv2Encoder(4, seed=0)
    with pytest.raises(ValueError):
        enc.encode_layer_activations({0: np.random.randn(3)})
