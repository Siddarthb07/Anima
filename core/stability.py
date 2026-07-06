"""Rolling readout stability metrics for intervention gating."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Protocol


class _ReadoutLike(Protocol):
    affect: dict[str, float]
    guard: Any


def _load_thresholds() -> dict[str, float]:
    import yaml

    p = Path(__file__).resolve().parent.parent / "probes" / "guard_config.yaml"
    defaults = {
        "stability_window": 8,
        "stability_score_threshold": 0.45,
        "valence_swing_threshold": 0.35,
        "stability_abstain_rate": 0.5,
    }
    if not p.exists():
        return defaults
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {**defaults, **data}


def _window_stats(
    valences: list[float],
    arousals: list[float],
    abstain: list[bool],
    end: int,
    window: int,
    th: dict[str, float],
) -> dict[str, float]:
    start = max(0, end - window + 1)
    v = valences[start : end + 1]
    a = arousals[start : end + 1]
    ab = abstain[start : end + 1]
    n = len(v)
    if n < 2:
        return {
            "valence_std": 0.0,
            "arousal_std": 0.0,
            "mean_abs_delta_valence": 0.0,
            "guard_abstain_rate": float(ab[-1]) if ab else 0.0,
            "stability_score": 1.0,
        }

    mean_v = sum(v) / n
    valence_std = math.sqrt(sum((x - mean_v) ** 2 for x in v) / n)
    mean_a = sum(a) / n
    arousal_std = math.sqrt(sum((x - mean_a) ** 2 for x in a) / n)
    deltas = [abs(v[i] - v[i - 1]) for i in range(1, len(v))]
    mean_abs_delta = sum(deltas) / len(deltas) if deltas else 0.0
    abstain_rate = sum(1 for x in ab if x) / n

    swing_th = float(th.get("valence_swing_threshold", 0.35))
    vol_penalty = min(1.0, valence_std / max(swing_th, 1e-6))
    delta_penalty = min(1.0, mean_abs_delta / max(swing_th, 1e-6))
    abstain_penalty = min(1.0, abstain_rate / max(float(th.get("stability_abstain_rate", 0.5)), 1e-6))
    stability_score = round(
        max(0.0, 1.0 - (0.45 * vol_penalty + 0.35 * delta_penalty + 0.20 * abstain_penalty)),
        4,
    )
    return {
        "valence_std": round(valence_std, 4),
        "arousal_std": round(arousal_std, 4),
        "mean_abs_delta_valence": round(mean_abs_delta, 4),
        "guard_abstain_rate": round(abstain_rate, 4),
        "stability_score": stability_score,
    }


def analyze_readout_stability(
    readouts: list[_ReadoutLike],
    *,
    window: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Per-token rolling stability flags + run-level summary.
    """
    th = _load_thresholds()
    win = int(window if window is not None else th.get("stability_window", 8))
    score_th = float(th.get("stability_score_threshold", 0.45))

    if not readouts:
        return [], {
            "stability_score": 1.0,
            "unstable_token_count": 0,
            "max_valence_swing": 0.0,
            "mean_valence_std": 0.0,
        }

    valences = [float(r.affect["valence"]) for r in readouts]
    arousals = [float(r.affect["arousal"]) for r in readouts]
    abstain = [bool(r.guard.abstain_recommended) for r in readouts]

    per_token: list[dict[str, Any]] = []
    unstable_count = 0
    max_swing = 0.0
    stds: list[float] = []

    for i in range(len(readouts)):
        stats = _window_stats(valences, arousals, abstain, i, win, th)
        unstable = stats["stability_score"] < score_th
        if unstable:
            unstable_count += 1
        max_swing = max(max_swing, stats["mean_abs_delta_valence"])
        stds.append(stats["valence_std"])
        per_token.append(
            {
                "unstable_window": unstable,
                "window_stability_score": stats["stability_score"],
                "window_valence_std": stats["valence_std"],
            }
        )

    run_score = round(sum(t["window_stability_score"] for t in per_token) / len(per_token), 4)
    summary = {
        "stability_score": run_score,
        "unstable_token_count": unstable_count,
        "max_valence_swing": round(max_swing, 4),
        "mean_valence_std": round(sum(stds) / len(stds), 4) if stds else 0.0,
        "stability_window": win,
    }
    return per_token, summary


def merge_stability_flags(readouts: list, per_token: list[dict[str, Any]]) -> None:
    """Attach stability bool flags to readout.flags in place."""
    for ro, pt in zip(readouts, per_token):
        ro.flags = {**ro.flags, "unstable_window": bool(pt.get("unstable_window", False))}


def apply_guard_gate(readouts: list, per_token: list[dict[str, Any]], *, guard_mode: str) -> int:
    """
    In gate mode, force high-uncertainty region for tokens in unstable windows.
    Returns count of gated tokens.
    """
    if guard_mode != "gate":
        return 0
    gated = 0
    for ro, pt in zip(readouts, per_token):
        if pt.get("unstable_window"):
            ro.region = "high-uncertainty"
            ro.region_analog = "No direct human analog — readout guard: unstable rolling window"
            ro.flags = {**ro.flags, "gated": True}
            gated += 1
    return gated
