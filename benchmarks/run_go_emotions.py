"""Tier B GoEmotions text probe benchmark (requires trained _text checkpoint)."""

from __future__ import annotations

import os


def run(model: str) -> dict:
    if os.environ.get("SKIP_GO_EMOTIONS") == "1":
        return {"tier": "external_text", "benchmark": "go_emotions", "status": "skipped"}
    try:
        from probes.train_text import build_goemotions_samples
        from core.extractor import ActivationExtractor
        from probes.linear_probe import AffectProbe
        from probes.zoo_io import load_probe_into, probe_slug
        import numpy as np

        ex = ActivationExtractor(model)
        probe = AffectProbe(ex.hidden_dim, len(ex.layer_indices))
        meta = load_probe_into(probe, model)
        if meta.get("probe_origin") == "random":
            ex.cleanup()
            return {
                "tier": "external_text",
                "benchmark": "go_emotions",
                "status": "skipped",
                "reason": "no_text_checkpoint",
            }
        acts, targets, _ = build_goemotions_samples(ex, split="validation", max_samples=200)
        pv, gv, pa, ga = [], [], [], []
        for i, a in enumerate(acts):
            o = probe.predict(a)
            pv.append(o["valence"])
            gv.append(targets[i]["valence"])
            pa.append(o["arousal"])
            ga.append(targets[i]["arousal"])
        rv = float(np.corrcoef(pv, gv)[0, 1]) if len(pv) > 3 else 0.0
        ra = float(np.corrcoef(pa, ga)[0, 1]) if len(pa) > 3 else 0.0
        ex.cleanup()
        return {
            "tier": "external_text",
            "benchmark": "go_emotions",
            "status": "ok",
            "pearson_valence": round(rv, 4),
            "pearson_arousal": round(ra, 4),
            "probe_origin": meta.get("probe_origin"),
        }
    except Exception as exc:
        return {"tier": "external_text", "benchmark": "go_emotions", "status": "error", "message": str(exc)}
