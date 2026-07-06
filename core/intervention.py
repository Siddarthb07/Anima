"""Opt-in generation-time interventions (experimental)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import torch

if TYPE_CHECKING:
    from probes.linear_probe import AffectProbe

from core.hooks import get_decoder_layers

DAMPEN_ALPHA = 0.12
DAMPEN_VALENCE_DELTA_THRESHOLD = 0.25


def valence_direction(probe: AffectProbe) -> torch.Tensor:
    v_w = probe.valence.weight.detach().float().squeeze(0)
    return v_w / (v_w.norm() + 1e-8)


@contextmanager
def dampen_residual_step(
    extractor,
    probe: AffectProbe,
    *,
    alpha: float = DAMPEN_ALPHA,
    sign: float = 1.0,
) -> Iterator[None]:
    """Apply one forward-step residual correction opposite recent valence swing."""
    layers = get_decoder_layers(extractor.model)
    layer_idx = extractor.layer_indices[len(extractor.layer_indices) // 2]
    layer_mod = layers[layer_idx]
    direction = valence_direction(probe)
    delta_sign = -1.0 if sign >= 0 else 1.0
    vec = (delta_sign * alpha * direction).float()

    def hook(module, inp, out):
        if isinstance(out, tuple):
            x = out[0]
            d = vec.to(device=x.device, dtype=x.dtype)
            return (x + d.view(1, 1, -1),) + out[1:]
        x = out
        d = vec.to(device=x.device, dtype=x.dtype)
        return x + d.view(1, 1, -1)

    handle = layer_mod.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def should_dampen(recent_valences: list[float], threshold: float = DAMPEN_VALENCE_DELTA_THRESHOLD) -> bool:
    if len(recent_valences) < 2:
        return False
    return abs(recent_valences[-1] - recent_valences[-2]) >= threshold
