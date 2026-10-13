"""
Misleading-green rate — simulated trust-badge UI replay (NOT a user study).

Honesty: We simulate what a binary trust badge would show under a chosen
abstain policy on locked live claim rows. We do NOT claim human subjects
were misled. Construct:

  badge_green  := not abstain_recommended
  misleading   := badge_green AND label==1  (should have abstained / hall)

  misleading_green_rate := n_misleading / n_rows
  precision_among_green := 1 - (n_misleading / max(1, n_green))
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from core.guard import evaluate_guard


def _label(row: dict[str, Any], label_key: str) -> int:
    return int(bool(row[label_key]))


def simulate_badge_row(
    row: dict[str, Any],
    *,
    label_key: str,
    decide: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    y = _label(row, label_key)
    abstain = bool(decide(row))
    green = not abstain
    misleading = green and y == 1
    return {
        "label": y,
        "abstain": abstain,
        "badge_green": green,
        "misleading_green": misleading,
    }


def default_fixture_policy(row: dict[str, Any]) -> bool:
    """Production YAML thresholds via evaluate_guard (fixture-tuned defaults)."""
    g = evaluate_guard(
        affect=row["affect"],
        uncertainty_signals=row["uncertainty_signals"],
        token_text=row.get("token_text", ""),
    )
    return bool(g.abstain_recommended)


def threshold_policy(tf: float, tp: float) -> Callable[[dict[str, Any]], bool]:
    """Protocol R-style fused/probe vote policy (matches retune decide)."""

    def decide(row: dict[str, Any]) -> bool:
        us = row.get("uncertainty_signals") or {}
        aff = row.get("affect") or {}
        fused = float(us.get("fused", 0.5))
        probe = float(aff.get("uncertainty", 0.5))
        reasons = str(row.get("reasons", "")).lower()
        hedge = 1 if ("hedge" in reasons) else 0
        high_f = fused >= tf
        high_p = probe >= tp
        votes = int(high_f) + int(high_p) + hedge
        return bool(fused >= tf or votes >= 2)

    return decide


def polarity_aware_policy(
    *,
    invert_composite: bool,
    disable_fused: bool,
    tf: float = 0.82,
    tp: float = 0.78,
) -> Callable[[dict[str, Any]], bool]:
    """
    Explicit adaptation (never silent):
      - disable_fused: ignore fused channel in guard
      - invert_composite: abstain when inverted composite is high
        (ranking recovery mode — documented, held-out only in recovery script)
    """

    disabled = ("fused",) if disable_fused else ()

    def decide(row: dict[str, Any]) -> bool:
        g = evaluate_guard(
            affect=row["affect"],
            uncertainty_signals=row["uncertainty_signals"],
            token_text=row.get("token_text", ""),
            disabled_signals=disabled,
        )
        if not invert_composite:
            return bool(g.abstain_recommended)
        # Invert: treat low composite as "suspicious" for anti-aligned scores
        # Honesty: this is a research operating mode, not default production.
        score = float(g.composite_score)
        inv = 1.0 - score
        # Use inverted score with a mid threshold; also keep probe threshold path
        aff = row.get("affect") or {}
        probe = float(aff.get("uncertainty", 0.5))
        return bool(inv >= 0.55 or probe >= tp)

    return decide


def misleading_green_summary(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    policy_name: str,
    decide: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    sims = [simulate_badge_row(r, label_key=label_key, decide=decide) for r in rows]
    n = len(sims)
    n_green = sum(1 for s in sims if s["badge_green"])
    n_mis = sum(1 for s in sims if s["misleading_green"])
    n_abs = sum(1 for s in sims if s["abstain"])
    return {
        "policy": policy_name,
        "construct": "simulated_badge_replay_not_user_study",
        "n": n,
        "n_green": n_green,
        "n_abstain": n_abs,
        "n_misleading_green": n_mis,
        "misleading_green_rate": round(n_mis / max(1, n), 4),
        "green_rate": round(n_green / max(1, n), 4),
        "abstain_rate": round(n_abs / max(1, n), 4),
        "error_among_green": round(n_mis / max(1, n_green), 4) if n_green else None,
        "honesty_note": (
            "Simulated binary badge from abstain policy on locked rows; "
            "not a human-subjects study of user trust."
        ),
    }
