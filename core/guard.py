"""Hallucination / readout reliability guard (multi-signal, abstain policy)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from probes.validate import hedge_score

# Signals that can be disabled for ablation studies.
ABLATION_SIGNALS = (
    "fused",
    "probe_uncertainty",
    "hedging",
)


@dataclass
class GuardDecision:
    tier: str  # HIGH | MEDIUM | LOW
    abstain_recommended: bool
    composite_score: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_thresholds() -> dict[str, float]:
    import yaml

    p = Path(__file__).resolve().parent.parent / "probes" / "guard_config.yaml"
    if not p.exists():
        return {
            "fused_high": 0.75,
            "fused_abstain": 0.82,
            "probe_uncertainty_abstain": 0.78,
            "hedge_words_threshold": 2,
        }
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def evaluate_guard(
    *,
    affect: dict[str, float],
    uncertainty_signals: dict[str, float],
    token_text: str = "",
    calibrator: Optional[Any] = None,
    abstain_regions: bool = True,
    disabled_signals: Optional[Sequence[str]] = None,
) -> GuardDecision:
    """
    Multi-signal abstain policy.

    ``disabled_signals`` (ablation): any of ``fused``, ``probe_uncertainty``, ``hedging``
    are ignored when listed — composite and reasons recompute without them.
    """
    th = _load_thresholds()
    disabled = {s.strip().lower() for s in (disabled_signals or ()) if s}

    def _finite(x: float, default: float = 0.5) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return default
        if v != v or v in (float("inf"), float("-inf")):
            return default
        return v

    fused_raw = uncertainty_signals.get("fused", 0.5)
    probe_raw = affect.get("uncertainty", 0.5)
    fused_ok = True
    try:
        fused_chk = float(fused_raw)
        fused_ok = fused_chk == fused_chk and fused_chk not in (float("inf"), float("-inf"))
    except (TypeError, ValueError):
        fused_ok = False
    probe_u = _finite(probe_raw, 0.5)
    reasons: list[str] = []

    fused = _finite(fused_raw, 0.5) if fused_ok else 0.5
    if calibrator is not None and hasattr(calibrator, "forward") and fused_ok:
        with torch.no_grad():
            raw = torch.tensor([fused], dtype=torch.float32)
            fused_cal = float(calibrator(raw).squeeze().item())
            if fused_cal != fused_cal:
                fused_ok = False
                fused_cal = 0.5
    else:
        fused_cal = fused if fused_ok else 0.5

    use_fused = "fused" not in disabled and fused_ok
    use_probe = "probe_uncertainty" not in disabled
    use_hedge = "hedging" not in disabled
    if not fused_ok:
        reasons.append("fused_channel_nan_dropped")

    # Reweight composite when ablating so remaining signals stay on [0,1].
    if use_fused and use_probe:
        composite = round(0.55 * fused_cal + 0.45 * probe_u, 4)
    elif use_fused:
        composite = round(fused_cal, 4)
    elif use_probe:
        composite = round(probe_u, 4)
    else:
        composite = 0.0

    hedge = hedge_score(token_text) if use_hedge else 0

    if use_fused and fused_cal >= th.get("fused_abstain", 0.82):
        reasons.append("high_fused_uncertainty")
    if use_probe and probe_u >= th.get("probe_uncertainty_abstain", 0.78):
        reasons.append("high_probe_uncertainty")
    if use_hedge and hedge >= th.get("hedge_words_threshold", 2):
        reasons.append("lexical_hedging")

    abstain_fused = use_fused and fused_cal >= th.get("fused_abstain", 0.82)
    abstain = len(reasons) >= 2 or abstain_fused

    if composite < 0.4:
        tier = "HIGH"
    elif composite < 0.7:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    if abstain and abstain_regions:
        reasons.append("suppress_strong_region_analog")

    return GuardDecision(
        tier=tier,
        abstain_recommended=abstain,
        composite_score=composite,
        reasons=reasons,
    )


def region_under_guard(
    valence: float,
    arousal: float,
    uncertainty: float,
    guard: GuardDecision,
) -> tuple[str, str]:
    """Apply label_region logic but force high-uncertainty branch when abstaining."""
    if guard.abstain_recommended:
        return (
            "high-uncertainty",
            "No direct human analog — readout guard: low internal confidence",
        )
    from core.regions import label_region

    return label_region(valence, arousal, uncertainty)
