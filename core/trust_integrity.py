"""
Trust Integrity Gate (TIG) — Green-Shield claim lock for Anima.

Novel contribution (methods / selective-prediction integrity):
  A fixture-perfect abstain UI can mint a permanently green trust badge while
  live claim-encoding ranking is chance or anti-aligned. TIG turns that failure
  mode into an *enforceable* claim bit:

    trust_ui_claim_allowed = false  until Protocol L (+ polarity + transfer gap)
    pass locked criteria. Same spirit as Corvex claim gates, for trust UI.

Protocol L′ (adapted) receipts may unlock when LOQO ridge fusion clears the
adapted bar. Baseline Protocol L receipts stay locked on current models —
that collapse finding must remain visible.

This module does NOT invent live AUROC; it consumes locked eval JSON and emits
a sealed receipt. Optional runtime enforce refuses "trusted" / green-tier
language when the bit is false.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# Baseline Protocol L unlock bar (deliberately hard — default composite FAILS).
DEFAULT_CRITERIA: dict[str, Any] = {
    "min_live_auroc": 0.55,
    "max_transfer_gap": 0.35,  # AUROC_F - AUROC_L
    "require_polarity_not_anti_aligned": True,
    "max_never_abstain_fraction": 0.95,
    "min_models": 1,
    "require_both_suites": True,
    "min_n_live": 40,
    "require_channel_sign_agree": True,
    "operating_point": "protocol_L_baseline",
}

# Adapted Protocol L′ unlock bar (LOQO ridge; HaluEval primary, TQA stress).
ADAPTED_CRITERIA: dict[str, Any] = {
    "min_live_auroc": 0.55,  # hard bar on primary suite (halueval)
    "max_transfer_gap": 0.35,  # same-scorer max(0, F' - L')
    "require_polarity_not_anti_aligned": True,
    "min_n_live": 40,
    "require_both_suites": True,
    "require_channel_sign_agree": False,  # single fused score under L′
    "operating_point": "protocol_L_prime_loqo",
    "primary_suite": "halueval",
    # TruthfulQA is stress: must not be anti-aligned; near-chance coherent OK
    "secondary_suite": "truthfulqa",
    "secondary_min_auroc": 0.45,
    "max_misleading_green_rate": 0.40,
}


@dataclass
class SuiteIntegrity:
    suite: str
    protocol_f_auroc: Optional[float]
    protocol_l_auroc: Optional[float]
    transfer_gap: Optional[float]  # F - L (or same-scorer for L′)
    auroc_ci95: Optional[dict[str, Any]]
    never_abstains: Optional[bool]
    n_live: Optional[int]
    polarity_verdict: Optional[str]
    probe_only_auroc: Optional[float]
    fused_only_auroc: Optional[float]
    channel_sign_conflict: Optional[bool]
    misleading_green_rate: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrustIntegrityReceipt:
    schema: str = "anima.trust_integrity.v1"
    generated_at: str = ""
    model: str = ""
    model_slug: str = ""
    trust_ui_claim_allowed: bool = False
    lab_verified: bool = False
    failed_gates: list[str] = field(default_factory=list)
    passed_gates: list[str] = field(default_factory=list)
    criteria: dict[str, Any] = field(default_factory=dict)
    suites: dict[str, Any] = field(default_factory=dict)
    transfer_gap_max: Optional[float] = None
    polarity_worst: Optional[str] = None
    seal_sha256: str = ""
    sources: list[str] = field(default_factory=list)
    operating_point: str = "protocol_L_baseline"
    innovation_note: str = (
        "Green-shield claim lock: fixture AUROC alone cannot unlock trust UI language."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def transfer_gap(auroc_f: float, auroc_l: float) -> float:
    """Protocol F AUROC minus Protocol L AUROC (positive = fixture optimism)."""
    return round(float(auroc_f) - float(auroc_l), 4)


def channel_sign_conflict(probe_auroc: Optional[float], fused_auroc: Optional[float]) -> Optional[bool]:
    if probe_auroc is None or fused_auroc is None:
        return None
    return bool((probe_auroc > 0.5) != (fused_auroc > 0.5))


def _sha256_canonical(obj: dict[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def evaluate_trust_integrity(
    *,
    model: str,
    model_slug: str,
    suites: dict[str, SuiteIntegrity],
    criteria: Optional[dict[str, Any]] = None,
    sources: Optional[list[str]] = None,
) -> TrustIntegrityReceipt:
    crit = {**DEFAULT_CRITERIA, **(criteria or {})}
    op = str(crit.get("operating_point") or "protocol_L_baseline")
    failed: list[str] = []
    passed: list[str] = []
    gaps: list[float] = []
    polarity_vals: list[str] = []

    if crit.get("require_both_suites") and set(suites.keys()) < {"halueval", "truthfulqa"}:
        failed.append("missing_suite")

    primary = str(crit.get("primary_suite") or "")
    secondary = str(crit.get("secondary_suite") or "")
    secondary_min = float(crit.get("secondary_min_auroc", crit["min_live_auroc"]))
    max_mgr = crit.get("max_misleading_green_rate")

    for name, s in suites.items():
        if s.n_live is not None and s.n_live < int(crit["min_n_live"]):
            failed.append(f"{name}:n_live<{crit['min_n_live']}")
        else:
            passed.append(f"{name}:n_live_ok")

        if s.protocol_l_auroc is None:
            failed.append(f"{name}:missing_live_auroc")
            continue

        # AUROC bar: primary uses hard min; secondary (L′ only) uses softer floor
        if op.startswith("protocol_L_prime") and secondary and name == secondary:
            min_a = secondary_min
        else:
            min_a = float(crit["min_live_auroc"])

        if s.protocol_l_auroc >= min_a:
            passed.append(f"{name}:live_auroc>={min_a}")
        else:
            failed.append(f"{name}:live_auroc={s.protocol_l_auroc}<{min_a}")

        if s.transfer_gap is not None:
            gaps.append(s.transfer_gap)
            # Under L′: hard transfer-gap bar applies to primary suite only.
            apply_gap = True
            if op.startswith("protocol_L_prime") and secondary and name == secondary:
                apply_gap = False
                passed.append(f"{name}:transfer_gap_advisory={s.transfer_gap}")
            if apply_gap:
                if s.transfer_gap <= float(crit["max_transfer_gap"]):
                    passed.append(f"{name}:transfer_gap<={crit['max_transfer_gap']}")
                else:
                    failed.append(
                        f"{name}:transfer_gap={s.transfer_gap}>{crit['max_transfer_gap']}"
                    )

        if s.never_abstains:
            failed.append(f"{name}:never_abstains_on_live")
        elif s.never_abstains is False:
            passed.append(f"{name}:some_abstain_events")

        if s.polarity_verdict:
            polarity_vals.append(s.polarity_verdict)
            if (
                crit.get("require_polarity_not_anti_aligned")
                and s.polarity_verdict == "anti_aligned_ranking"
            ):
                failed.append(f"{name}:anti_aligned_ranking")
            else:
                passed.append(f"{name}:polarity_ok")

        if crit.get("require_channel_sign_agree"):
            if s.channel_sign_conflict:
                failed.append(f"{name}:channel_sign_conflict")
            elif s.channel_sign_conflict is False:
                passed.append(f"{name}:channels_agree_side")

        if max_mgr is not None and s.misleading_green_rate is not None:
            if s.misleading_green_rate <= float(max_mgr):
                passed.append(f"{name}:misleading_green<={max_mgr}")
            else:
                failed.append(
                    f"{name}:misleading_green={s.misleading_green_rate}>{max_mgr}"
                )

    allowed = len(failed) == 0
    receipt = TrustIntegrityReceipt(
        generated_at=datetime.now(timezone.utc).isoformat(),
        model=model,
        model_slug=model_slug,
        trust_ui_claim_allowed=allowed,
        lab_verified=True,
        failed_gates=failed,
        passed_gates=passed,
        criteria=crit,
        suites={k: v.to_dict() for k, v in suites.items()},
        transfer_gap_max=max(gaps) if gaps else None,
        polarity_worst=(
            "anti_aligned_ranking"
            if "anti_aligned_ranking" in polarity_vals
            else (polarity_vals[0] if polarity_vals else None)
        ),
        sources=list(sources or []),
        operating_point=op,
        innovation_note=(
            "Green-shield claim lock: fixture AUROC alone cannot unlock trust UI. "
            + (
                "L′ unlock uses LOQO ridge fusion; HaluEval is the primary suite."
                if op.startswith("protocol_L_prime")
                else "Baseline L uses default composite scores."
            )
        ),
    )
    body = receipt.to_dict()
    body.pop("seal_sha256", None)
    receipt.seal_sha256 = _sha256_canonical(body)
    return receipt


def enforce_trust_language(
    *,
    claim_allowed: bool,
    proposed_tier: str,
    abstain_recommended: bool,
) -> dict[str, Any]:
    if claim_allowed:
        return {
            "trust_ui_claim_allowed": True,
            "tier": proposed_tier,
            "abstain_recommended": abstain_recommended,
            "badge": "allowed",
            "ui_message": None,
        }
    return {
        "trust_ui_claim_allowed": False,
        "tier": "LOW",
        "abstain_recommended": True,
        "badge": "locked",
        "ui_message": (
            "Trust UI locked: live Protocol L / polarity / transfer-gap gates not met. "
            "Fixture AUROC alone cannot unlock a green shield."
        ),
    }


def load_receipt(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_receipt(receipt: TrustIntegrityReceipt, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt.to_dict(), indent=2), encoding="utf-8")
    return path
