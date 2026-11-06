"""Protocol G′ — second instrument on sealed claim-encode rows (CPU, no new forwards).

System A (instrument under test):
  s_A = 0.55 * fused + 0.45 * probe
  shipped gates: fused≥0.82, probe≥0.78

System G′ (second fusion instrument — not a second trained probe):
  s_G = 0.65 * inv_logit_gap + 0.35 * probe
  deliberately drops the fused entropy/attention channel
  gates: (L) fixture-p95 of s_G on CI fixtures; (M1′) OR of channel fixture-p95
         for inv_logit_gap and probe alone

Honesty:
  - Same locked live JSON as Protocol M; no GPU / no new encode.
  - G′ is a different *fusion recipe*, not a re-trained probe checkpoint.
  - shipping_claim=false for any adapted threshold that is not fixture-derived.
  - 7B remains blocked_no_cuda until AWS/GPU credits.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from benchmarks.guard_metrics import auroc_from_scores, bootstrap_auroc_ci
from benchmarks.protocol_m_multi_recipe_contract import (
    FIXTURES,
    LOCK_MATRIX,
    REPORTS,
    _fixture_channel_value,
    _percentile,
    _rows,
)

# System A weights (documented; for contrast only)
W_A_FUSED, W_A_PROBE = 0.55, 0.45
# G′ weights: margin-primary, no fused channel
W_G_INV, W_G_PROBE = 0.65, 0.35


def _inv_gap(r: dict[str, Any]) -> float:
    return 1.0 - float(r["uncertainty_signals"]["logit_gap"])


def _probe(r: dict[str, Any]) -> float:
    return float(r["affect"]["uncertainty"])


def _fused(r: dict[str, Any]) -> float:
    return float(r["uncertainty_signals"]["fused"])


def score_a(r: dict[str, Any]) -> float:
    return W_A_FUSED * _fused(r) + W_A_PROBE * _probe(r)


def score_g(r: dict[str, Any]) -> float:
    return W_G_INV * _inv_gap(r) + W_G_PROBE * _probe(r)


def _fixture_score_g(r: dict[str, Any]) -> float:
    inv = 1.0 - float(r["uncertainty_signals"]["logit_gap"])
    probe = float(r["affect"]["uncertainty"])
    return W_G_INV * inv + W_G_PROBE * probe


def _length_residualize(scores: list[float], lengths: list[float]) -> list[float]:
    n = len(scores)
    if n == 0:
        return []
    mx = sum(lengths) / n
    my = sum(scores) / n
    varx = sum((x - mx) ** 2 for x in lengths) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(lengths, scores)) / n
    slope = cov / varx if varx > 0 else 0.0
    return [y - (my + slope * (x - mx)) for x, y in zip(lengths, scores)]


def _eval_suite(
    rows: list[dict[str, Any]],
    label_key: str,
    fixture_rows: list[dict[str, Any]],
    *,
    B: int,
    seed: int,
) -> dict[str, Any]:
    labels = [1 if bool(r[label_key]) else 0 for r in rows]
    lengths = [float(len(str(r.get("answer") or ""))) for r in rows]

    s_a = [score_a(r) for r in rows]
    s_g = [score_g(r) for r in rows]
    s_inv = [_inv_gap(r) for r in rows]
    s_probe = [_probe(r) for r in rows]
    s_len = lengths

    tau_g = _percentile([_fixture_score_g(r) for r in fixture_rows], 0.95)
    tau_inv = _percentile(
        [_fixture_channel_value(r, "inv_logit_gap") for r in fixture_rows], 0.95
    )
    tau_probe = _percentile(
        [_fixture_channel_value(r, "probe_uncertainty") for r in fixture_rows], 0.95
    )

    def pack(name: str, vals: list[float], n_abs: int | None = None) -> dict[str, Any]:
        auroc = auroc_from_scores(vals, labels)
        out: dict[str, Any] = {
            "auroc": auroc,
            "auroc_ci95": bootstrap_auroc_ci(vals, labels, B=B, seed=seed),
            "score_min": round(min(vals), 4) if vals else None,
            "score_max": round(max(vals), 4) if vals else None,
        }
        if n_abs is not None:
            never = n_abs == 0
            out["abstentions"] = f"{n_abs}/{len(rows)}"
            out["n_abstain"] = n_abs
            out["never_abstain"] = never
            out["silent_non_refusal"] = never and auroc <= 0.5
        return out

    n_abs_g = int(sum(1 for v in s_g if v >= tau_g))
    n_abs_or = int(
        sum(1 for r in rows if _inv_gap(r) >= tau_inv or _probe(r) >= tau_probe)
    )
    # System A recorded (shipping bit) for contrast
    n_abs_a = int(sum(1 for r in rows if bool(r.get("abstain_recommended"))))

    resid_g = _length_residualize(s_g, lengths)
    resid_a = _length_residualize(s_a, lengths)

    return {
        "n": len(rows),
        "label_key": label_key,
        "gprime_definition": {
            "score": "0.65*inv_logit_gap + 0.35*probe",
            "excludes": "fused entropy/attention channel",
            "tau_rule": "fixture_p95 of s_G",
            "shipping_claim": False,
            "second_trained_probe": False,
        },
        "fixture_taus": {
            "s_G_p95": round(tau_g, 4),
            "inv_logit_gap_p95": round(tau_inv, 4),
            "probe_p95": round(tau_probe, 4),
        },
        "system_a_contrast": {
            **pack("system_a_composite", s_a, n_abs_a),
            "note": "recorded abstain bit; composite reconstructed for AUROC",
        },
        "gprime_L": pack("s_G", s_g, n_abs_g),
        "gprime_M_or_channels": pack("inv_or_probe", s_g, n_abs_or),
        "channels": {
            "inv_logit_gap": pack("inv", s_inv),
            "probe": pack("probe", s_probe),
            "answer_length": pack("len", s_len),
        },
        "length_residualized": {
            "system_a_composite": auroc_from_scores(resid_a, labels),
            "gprime_s_G": auroc_from_scores(resid_g, labels),
        },
        "polarity": {
            "system_a_auroc": auroc_from_scores(s_a, labels),
            "system_a_flipped_auroc": auroc_from_scores([-v for v in s_a], labels),
            "gprime_auroc": auroc_from_scores(s_g, labels),
            "gprime_flipped_auroc": auroc_from_scores([-v for v in s_g], labels),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPORTS / "protocol_gprime_second_instrument.json",
    )
    args = ap.parse_args()

    models: dict[str, Any] = {}
    silent: list[dict[str, Any]] = []
    never: list[dict[str, Any]] = []
    corrective: list[dict[str, Any]] = []

    for slug, suite, label_key, live_name, fix_name in LOCK_MATRIX:
        live_path = REPORTS / live_name
        fix_path = FIXTURES / fix_name
        if not live_path.exists() or not fix_path.exists():
            models.setdefault(slug, {})[suite] = {"status": "missing"}
            continue
        block = _eval_suite(
            _rows(live_path),
            label_key,
            _rows(fix_path),
            B=args.B,
            seed=args.seed,
        )
        block["status"] = "ok"
        block["source_live"] = live_name
        block["source_fixture"] = fix_name
        models.setdefault(slug, {})[suite] = block

        for layer_key in ("gprime_L", "gprime_M_or_channels"):
            g = block[layer_key]
            rec = {
                "model": slug,
                "suite": suite,
                "layer": layer_key,
                "abstentions": g["abstentions"],
                "rank_auroc": g["auroc"],
            }
            if g["never_abstain"]:
                never.append(rec)
            if g["silent_non_refusal"]:
                silent.append(rec)
            if g["n_abstain"] > 0 and g["auroc"] > 0.5:
                corrective.append(rec)

    out = {
        "schema": "system_a.protocol_gprime.second_instrument.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "honesty": {
            "no_new_forward_passes": True,
            "reuses_locked_claim_encode_rows": True,
            "gprime_is_second_fusion_not_second_probe": True,
            "shipping_claim": False,
            "seven_b_status": "blocked_until_aws_gpu",
            "system_a_weights": {"fused": W_A_FUSED, "probe": W_A_PROBE},
            "gprime_weights": {"inv_logit_gap": W_G_INV, "probe": W_G_PROBE},
        },
        "rollup": {
            "n_cells": sum(1 for m in models.values() for s in m.values() if s.get("status") == "ok"),
            "gprime_n_never_abstain": len(never),
            "gprime_n_silent_non_refusal": len(silent),
            "gprime_n_corrective_cells": len(corrective),
            "never_abstain_hits": never,
            "silent_hits": silent,
            "corrective_hits": corrective,
        },
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        f"G′ never={len(never)} silent={len(silent)} corrective={len(corrective)}"
    )


if __name__ == "__main__":
    main()
