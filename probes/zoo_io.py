"""Load/save probe checkpoints with sidecar metadata and optional calibrators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch

from probes.linear_probe import AffectProbe

ZOO_DIR = Path(__file__).resolve().parent / "zoo"


def probe_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower().replace("-", "_")


def checkpoint_path(slug: str, suffix: str = "") -> Path:
    name = f"{slug}{suffix}.pt"
    return ZOO_DIR / name


def meta_path(slug: str, suffix: str = "") -> Path:
    stem = f"{slug}{suffix}"
    return ZOO_DIR / f"{stem}.meta.json"


def calib_path(slug: str, suffix: str = "") -> Path:
    stem = f"{slug}{suffix}"
    return ZOO_DIR / f"{stem}.calib.pt"


def tribe_weights_path(slug: str) -> Path:
    return ZOO_DIR / f"{slug}_tribe_proj.npz"


def load_meta(slug: str, suffix: str = "") -> dict[str, Any]:
    p = meta_path(slug, suffix)
    if not p.exists():
        return {"probe_origin": "random", "slug": slug}
    return json.loads(p.read_text(encoding="utf-8"))


def save_meta(slug: str, meta: dict[str, Any], suffix: str = "") -> Path:
    ZOO_DIR.mkdir(parents=True, exist_ok=True)
    p = meta_path(slug, suffix)
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return p


def load_brain_meta(model_name: str) -> dict[str, Any]:
    """Sidecar meta for the brain (_narratives_pca) checkpoint, if present."""
    slug = probe_slug(model_name)
    if meta_path(slug, "_narratives_pca").exists():
        return load_meta(slug, "_narratives_pca")
    return {}


def brain_data_tier(probe_origin: str) -> str:
    """Research-grade label for UI and /models."""
    if probe_origin == "narratives_fMRI":
        return "real_fMRI"
    if probe_origin == "narratives_fMRI_synthetic_minimal":
        return "synthetic_minimal"
    return "none"


def _prefer_text_probe() -> bool:
    """Portfolio/public demos use text-emotion probes, not synthetic brain checkpoints."""
    import os

    from core.limits import public_mode_enabled

    if public_mode_enabled():
        return True
    return os.environ.get("ANIMA_PREFER_TEXT_PROBE", "").strip().lower() in ("1", "true", "yes")


def resolve_checkpoint_slug(model_name: str) -> tuple[Optional[Path], str, dict[str, Any]]:
    """Pick best zoo checkpoint: narratives > text > base slug (text first when prefer-text/public)."""
    slug = probe_slug(model_name)
    suffix_order = ("_text", "_narratives_pca", "") if _prefer_text_probe() else ("_narratives_pca", "_text", "")
    for suffix in suffix_order:
        ckpt = checkpoint_path(slug, suffix)
        if ckpt.exists():
            meta = load_meta(slug, suffix)
            meta.setdefault("probe_origin", "zoo")
            meta["checkpoint_suffix"] = suffix
            return ckpt, slug, meta
    return None, slug, {"probe_origin": "random", "slug": slug}


def load_probe_into(
    probe: AffectProbe,
    model_name: str,
    *,
    map_location: str = "cpu",
) -> dict[str, Any]:
    ckpt, slug, meta = resolve_checkpoint_slug(model_name)
    if ckpt is not None:
        state = torch.load(ckpt, map_location=map_location, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            probe.load_state_dict(state["state_dict"])
            meta = {**meta, **{k: v for k, v in state.items() if k != "state_dict"}}
        else:
            probe.load_state_dict(state)
        meta["checkpoint"] = str(ckpt)
    else:
        meta["checkpoint"] = None
    probe.eval()
    return meta


def save_probe_bundle(
    probe: AffectProbe,
    slug: str,
    meta: dict[str, Any],
    *,
    suffix: str = "",
) -> Path:
    ZOO_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": probe.state_dict(),
        **{k: v for k, v in meta.items() if k != "state_dict"},
    }
    out = checkpoint_path(slug, suffix)
    torch.save(bundle, out)
    save_meta(slug, meta, suffix=suffix)
    return out
