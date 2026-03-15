"""Tier A Narratives holdout encoding benchmark (requires NARRATIVES_ROOT)."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _resolve_root() -> str:
    root = os.environ.get("NARRATIVES_ROOT", "")
    if root and Path(root).exists():
        return root
    fallback = Path(__file__).resolve().parent.parent / "data" / "narratives_minimal"
    if fallback.is_dir():
        return str(fallback)
    return ""


def run(model: str) -> dict:
    root = _resolve_root()
    if not root:
        return {
            "tier": "external",
            "benchmark": "narratives_holdout",
            "status": "skipped",
            "reason": "NARRATIVES_ROOT unset or missing",
        }
    try:
        from core.extractor import ActivationExtractor
        from probes.train import DEFAULT_HOLDOUT, DEFAULT_TRAIN_STORIES, build_training_set, _eval_probe
        from probes.linear_probe import AffectProbe
        from probes.zoo_io import load_probe_into, probe_slug, save_probe_bundle

        ex = ActivationExtractor(model)
        probe = AffectProbe(ex.hidden_dim, len(ex.layer_indices))
        load_probe_into(probe, model)
        split_path = Path(__file__).parent / "splits" / "narratives_holdout.json"
        if split_path.exists():
            spec = json.loads(split_path.read_text(encoding="utf-8"))
            train_stories = spec.get("train", DEFAULT_TRAIN_STORIES)
            holdout = spec.get("holdout", DEFAULT_HOLDOUT)
        else:
            train_stories, holdout = DEFAULT_TRAIN_STORIES, DEFAULT_HOLDOUT
        tr_a, tr_v, tr_av, tr_u, va, vv, va_a, vu, meta = build_training_set(
            ex, root, train_stories, holdout
        )
        metrics = _eval_probe(probe, va, vv, va_a)
        ex.cleanup()
        return {
            "tier": "external",
            "benchmark": "narratives_holdout",
            "status": "ok",
            **metrics,
            "baselines": meta.get("baselines", {}),
            "holdout_stories": holdout,
        }
    except Exception as exc:
        return {"tier": "external", "benchmark": "narratives_holdout", "status": "error", "message": str(exc)}
