"""
Construct supervised targets from fMRI without inventing anatomy.

Default: PCA axes on training timepoints (no ROI labels in UI copy).
Optional: user-supplied voxel index arrays per ROI (numpy archives).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


class PCATargetTransformer:
    """Fit PCA on training BOLD (TR × voxels), expose valence/arousal proxies."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self._pca = None
        self._v_scale = 1.0
        self._a_min = 0.0
        self._a_max = 1.0

    def fit(self, Y_train: np.ndarray) -> "PCATargetTransformer":
        from sklearn.decomposition import PCA

        self._pca = PCA(n_components=min(self.n_components, Y_train.shape[1], Y_train.shape[0]))
        Z = self._pca.fit_transform(Y_train)
        self._v_scale = float(np.std(Z[:, 0]) + 1e-6)
        if Z.shape[1] > 1:
            a = Z[:, 1]
            self._a_min = float(a.min())
            self._a_max = float(a.max())
        else:
            self._a_min = 0.0
            self._a_max = 1.0
        return self

    def transform(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._pca is None:
            raise RuntimeError("PCATargetTransformer.fit must be called first")
        Z = self._pca.transform(Y)
        valence = np.clip(Z[:, 0] / self._v_scale, -1.0, 1.0)
        if Z.shape[1] > 1:
            arousal = (Z[:, 1] - self._a_min) / (self._a_max - self._a_min + 1e-9)
            arousal = np.clip(arousal, 0.0, 1.0)
        else:
            arousal = np.full_like(valence, 0.5)
        return valence.astype(np.float64), arousal.astype(np.float64)


def load_roi_index_npz(path: str) -> dict[str, np.ndarray]:
    """
    Load ROI voxel indices aligned to the same voxel ordering as preprocessed BOLD.

    File format: numpy savez with arrays amygdala=np.array([...]), acc=..., vmpfc=...
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    data = np.load(p, allow_pickle=False)
    return {k: data[k] for k in data.files}


def atlas_roi_means_row(Y_row: np.ndarray, roi_indices: dict[str, np.ndarray]) -> dict[str, float]:
    out = {}
    for name, idx in roi_indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < Y_row.shape[0])]
        out[name] = float(Y_row[idx].mean()) if idx.size else 0.0
    return out


def roi_dict_to_valence_arousal(means: dict[str, float]) -> tuple[float, float]:
    """Literature-inspired mixing — only meaningful if ROI indices are valid."""
    amy = float(means.get("amygdala", 0.0))
    vmpfc = float(means.get("vmpfc", 0.0))
    acc = float(means.get("acc", 0.0))
    valence = float(np.clip((vmpfc - amy) / 2.0, -1.0, 1.0))
    arousal = float(np.clip((acc + 1.0) / 2.0, 0.0, 1.0))
    return valence, arousal
