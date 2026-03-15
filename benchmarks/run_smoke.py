"""Tier C internal smoke benchmark."""

from __future__ import annotations

from core.defaults import DEFAULT_CAUSAL_LM
from core.extractor import ActivationExtractor
from probes.zoo_io import load_probe_into
from probes.linear_probe import AffectProbe


def run(model: str = DEFAULT_CAUSAL_LM) -> dict:
    ex = ActivationExtractor(model)
    probe = AffectProbe(ex.hidden_dim, len(ex.layer_indices))
    meta = load_probe_into(probe, model)
    rows = ex.extract("Hello", max_new_tokens=4)
    ex.cleanup()
    return {
        "tier": "internal",
        "benchmark": "smoke_extract",
        "n_tokens": len(rows),
        "probe_origin": meta.get("probe_origin", "random"),
        "status": "ok" if rows else "fail",
    }
