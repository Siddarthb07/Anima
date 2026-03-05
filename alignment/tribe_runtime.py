"""Optional Meta TRIBEv2 runtime adapter (heavy dependency)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from alignment.tribe_encoder import ROI_DEFINITIONS

_CACHE_ROOT = Path(__file__).resolve().parent.parent / "data" / "tribe_cache"


def tribe_runtime_available() -> bool:
    try:
        import tribev2  # noqa: F401

        return True
    except ImportError:
        return False


def predict_text_roi_summary(
    text: str,
    *,
    cache_key: Optional[str] = None,
) -> Optional[dict[str, float]]:
    """
    Call TRIBEv2 when installed; map cortical output to our 5 ROI scalars.
    Returns None if runtime unavailable or prediction fails.
    """
    if not tribe_runtime_available():
        return None
    if cache_key:
        cache_path = _CACHE_ROOT / f"{cache_key}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            return {k: float(data[k]) for k in ROI_DEFINITIONS if k in data}

    try:
        # Lazy integration point — actual API depends on tribev2 package layout.
        from tribev2 import TribeModel  # type: ignore

        model = TribeModel.from_pretrained("facebook/tribev2")
        # Placeholder: users with GPU run full pipeline; we store mean parcel proxies.
        _ = model
        _ = text
        # Until full vertex→ROI parcellation ships, return None to fall back to surrogate.
        return None
    except Exception:
        return None


def get_tribe_mode() -> str:
    return os.environ.get("ANIMA_TRIBE_MODE", "surrogate").lower()
