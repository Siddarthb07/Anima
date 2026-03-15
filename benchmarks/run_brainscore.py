"""Tier A Brain-Score Language (optional; requires brainscore-language + Py3.11)."""

from __future__ import annotations

import os


def run(model: str, benchmark_id: str = "Futrell2018-pearsonr") -> dict:
    if os.environ.get("SKIP_BRAINSCORE") == "1":
        return {"tier": "external", "benchmark": "brainscore_language", "status": "skipped"}
    try:
        from brainscore_language import score

        result = score(model_identifier=model, benchmark_identifier=benchmark_id)
        return {
            "tier": "external",
            "benchmark": "brainscore_language",
            "benchmark_id": benchmark_id,
            "status": "ok",
            "score": float(result),
        }
    except ImportError:
        return {
            "tier": "external",
            "benchmark": "brainscore_language",
            "status": "skipped",
            "reason": "brainscore_language not installed",
        }
    except Exception as exc:
        return {"tier": "external", "benchmark": "brainscore_language", "status": "error", "message": str(exc)}
