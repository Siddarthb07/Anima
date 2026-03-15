"""Compare Anima TRIBE surrogate vs optional tribev2 runtime on story text."""

from __future__ import annotations

from typing import Any

from alignment.tribe_runtime import get_tribe_mode, predict_text_roi_summary, tribe_runtime_available


def run(model: str, *, sample_text: str = "The man walked through the tunnel.") -> dict[str, Any]:
    mode = get_tribe_mode()
    runtime_ok = tribe_runtime_available()
    runtime_scores = predict_text_roi_summary(sample_text, cache_key="benchmark_smoke")
    return {
        "benchmark": "tribe_reference",
        "tier": "external",
        "status": "ok" if runtime_ok else "skipped",
        "model": model,
        "anima_tribe_mode": mode,
        "runtime_available": runtime_ok,
        "runtime_roi_scores": runtime_scores,
        "reason": None if runtime_ok else "tribev2 not installed — surrogate-only CI path",
    }
