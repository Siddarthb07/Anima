"""TRIBEv2-style ROI pathway driven by live LM geometry (surrogate projections).

End-to-end in anima means: probed-layer hidden states -> fixed seeded projections ->
named ROI scalars -> VA aggregate comparable to the legacy heuristic mapper.

This is **not** a voxel-level TRIBE decoder trained on fMRI; Narratives/atlas training
remains in ``probes/train.py``. Without atlas-specific weights, we expose ROI-aligned
axes derived deterministically from hidden vectors so the dashboard pipeline stays
consistent and inspectable.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np

ROI_DEFINITIONS: Dict[str, dict] = {
    "tpj": {"description": "Temporo-parietal junction — social / emotional salience (surrogate axis)"},
    "amygdala": {"description": "Amygdala-like axis — threat vs reward contrast proxy"},
    "acc": {"description": "Anterior cingulate-like axis — arousal / conflict proxy"},
    "vmpfc": {"description": "Ventromedial PFC-like axis — positive value proxy"},
    "broca": {"description": "Broca-like axis — local linguistic structure proxy"},
}

TRIBEv2_SURROGATE_NOTE = (
    "Surrogate TRIBEv2 ROI path: tanh-normalized dot products of mean probed-layer "
    "hidden states against seeded unit axes per ROI (per-model seed). Same LM tensors "
    "as affect probes; not TRIBE voxel reconstructions."
)


def tribe_seed(model_name: str) -> int:
    digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def _hidden_to_numpy(h: Any) -> np.ndarray:
    """Accept HF torch tensors or NumPy arrays without importing torch at module import time."""
    if hasattr(h, "detach") and callable(getattr(h, "detach", None)):
        try:
            import torch

            if isinstance(h, torch.Tensor):
                x = h.detach().float().cpu().numpy()
                return np.asarray(x, dtype=np.float64).reshape(-1)
        except ImportError:
            pass
    arr = np.asarray(h, dtype=np.float64)
    return arr.reshape(-1)


class TRIBEv2Encoder:
    """Maps pooled LM hidden states to named ROI surrogate scores."""

    def __init__(self, hidden_dim: int, seed: int = 42):
        self.hidden_dim = int(hidden_dim)
        self.available = self.hidden_dim > 0
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed & 0xFFFFFFFF)
        self._weights: dict[str, np.ndarray] = {}
        for roi in ROI_DEFINITIONS:
            w = rng.standard_normal(self.hidden_dim).astype(np.float64)
            w /= np.linalg.norm(w) + 1e-12
            self._weights[roi] = w

    def encode_layer_activations(self, activations: Dict[int, Any]) -> dict[str, float]:
        """Average ROI scores across all provided layers (same set probed by AffectProbe)."""
        if not activations:
            return {roi: 0.0 for roi in ROI_DEFINITIONS}
        acc = {roi: 0.0 for roi in ROI_DEFINITIONS}
        n = 0
        for _, h in activations.items():
            part = self._encode_hidden_vector(h)
            for roi in acc:
                acc[roi] += part[roi]
            n += 1
        return {roi: round(acc[roi] / n, 4) for roi in acc}

    def _encode_hidden_vector(self, h: Any) -> dict[str, float]:
        x = _hidden_to_numpy(h)
        if x.shape[0] != self.hidden_dim:
            raise ValueError(f"Hidden dim {x.shape[0]} does not match encoder {self.hidden_dim}")
        scale = 1.0 / np.sqrt(float(self.hidden_dim))
        return {roi: round(float(np.tanh(np.dot(self._weights[roi], x) * scale)), 4) for roi in ROI_DEFINITIONS}

    def derived_va_from_rois(self, roi_scores: dict[str, float]) -> dict[str, float]:
        """Collapse surrogate ROI scalars to a 2D valence/arousal sketch (same recipe as legacy stub)."""
        amygdala = float(roi_scores.get("amygdala", 0.0))
        vmpfc = float(roi_scores.get("vmpfc", 0.0))
        acc = float(roi_scores.get("acc", 0.0))
        valence = vmpfc - amygdala
        arousal = acc
        valence_norm = max(-1.0, min(1.0, valence / 2.0))
        arousal_norm = max(0.0, min(1.0, (arousal + 1.0) / 2.0))
        return {"valence": round(valence_norm, 4), "arousal": round(arousal_norm, 4)}

    def encode_text(self, text: str) -> dict:
        """Offline helper — LM states required for encoding; API uses encode_layer_activations."""
        _ = text
        return {roi: None for roi in ROI_DEFINITIONS}

    def to_probe_targets(self, roi_activations: dict) -> Optional[dict]:
        """Legacy API compatibility."""
        if any(v is None for v in roi_activations.values()):
            return None
        roi_scores = {k: float(v) for k, v in roi_activations.items() if isinstance(v, (int, float))}
        if len(roi_scores) != len(ROI_DEFINITIONS):
            return None
        return self.derived_va_from_rois(roi_scores)
