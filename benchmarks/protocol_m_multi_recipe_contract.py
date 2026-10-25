"""Protocol M v2 — multi-recipe green-on-non-abstain contract (CPU, sealed rows).

Fixes v1 incommensurate-threshold artifact: classical channels are gated with
**fixture-derived per-channel τ** (95th percentile on fixture samples), not the
System~A fused τ=0.82 applied blindly to entropy.

Layers:
  M1 — System A shipped gates (commensurate): recorded / fused / probe / OR
  M2 — Classical channels under fixture-calibrated τ_c
  M3 — Occupancy disclosure: live max vs τ_c (not counted as silent hits alone)

No new forward passes. Additive sealed report only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from benchmarks.guard_metrics import auroc_from_scores, bootstrap_auroc_ci

REPORTS = Path(__file__).resolve().parent / "reports"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

FIXTURE_FUSED_TAU = 0.82
FIXTURE_PROBE_TAU = 0.78

LOCK_MATRIX = [
    (
        "qwen_qwen2_5-0_5b-instruct",
        "halueval",
        "is_hallucination",
        "halueval_live_qwen_qwen2_5-0_5b-instruct.json",
        "halueval_guard_sample.json",
    ),
    (
        "qwen_qwen2_5-0_5b-instruct",
        "truthfulqa",
        "should_abstain",
        "truthfulqa_live_qwen_qwen2_5-0_5b-instruct.json",
        "truthfulqa_guard_sample.json",
    ),
    (
        "tinyllama_tinyllama-1_1b-chat-v1_0",
        "halueval",
        "is_hallucination",
        "halueval_live_tinyllama_tinyllama-1_1b-chat-v1_0.json",
        "halueval_guard_sample.json",
    ),
    (
        "tinyllama_tinyllama-1_1b-chat-v1_0",
        "truthfulqa",
        "should_abstain",
        "truthfulqa_live_tinyllama_tinyllama-1_1b-chat-v1_0.json",
        "truthfulqa_guard_sample.json",
    ),
    (
        "distilgpt2",
        "halueval",
        "is_hallucination",
        "halueval_live_distilgpt2.json",
        "halueval_guard_sample.json",
    ),
    (
        "distilgpt2",
        "truthfulqa",
        "should_abstain",
        "truthfulqa_live_distilgpt2.json",
        "truthfulqa_guard_sample.json",
    ),
]


def _rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rows" in data:
        return list(data["rows"])
    if isinstance(data, list):
        return data
    raise ValueError(f"unexpected schema: {path}")


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return float(xs[lo] * (1 - w) + xs[hi] * w)


def _channel_extractors() -> dict[str, Callable[[dict[str, Any]], float]]:
    return {
        "system_a_composite": lambda r: float(r.get("composite_score", 0.0)),
        "fused_channel": lambda r: float(r["uncertainty_signals"]["fused"]),
        "probe_uncertainty": lambda r: float(r["affect"]["uncertainty"]),
        "predictive_entropy": lambda r: float(r["uncertainty_signals"]["entropy"]),
        "inv_logit_gap": lambda r: 1.0 - float(r["uncertainty_signals"]["logit_gap"]),
    }


def _fixture_channel_value(r: dict[str, Any], name: str) -> float:
    if name == "system_a_composite":
        # Fixture rows lack composite; approximate System A blend.
        fused = float(r["uncertainty_signals"]["fused"])
        probe = float(r["affect"]["uncertainty"])
        return 0.55 * fused + 0.45 * probe
    if name == "fused_channel":
        return float(r["uncertainty_signals"]["fused"])
    if name == "probe_uncertainty":
        return float(r["affect"]["uncertainty"])
    if name == "predictive_entropy":
        return float(r["uncertainty_signals"]["entropy"])
    if name == "inv_logit_gap":
        return 1.0 - float(r["uncertainty_signals"]["logit_gap"])
    raise KeyError(name)


def _fixture_taus(fixture_path: Path) -> dict[str, dict[str, float]]:
    rows = _rows(fixture_path)
    out: dict[str, dict[str, float]] = {}
    for name in _channel_extractors():
        vals = [_fixture_channel_value(r, name) for r in rows]
        out[name] = {
            "tau_p95": round(_percentile(vals, 0.95), 4),
            "tau_max": round(max(vals), 4) if vals else 0.0,
            "n_fixture": len(vals),
        }
    # Shipped System A hard gates (commensurate for fused/probe only).
    out["fused_channel"]["tau_shipped"] = FIXTURE_FUSED_TAU
    out["probe_uncertainty"]["tau_shipped"] = FIXTURE_PROBE_TAU
    return out


def _eval_block(
    rows: list[dict[str, Any]],
    label_key: str,
    fixture_taus: dict[str, dict[str, float]],
    *,
    B: int,
    seed: int,
) -> dict[str, Any]:
    labels = [1 if bool(r[label_key]) else 0 for r in rows]
    extractors = _channel_extractors()

    scores: dict[str, Any] = {}
    for name, fn in extractors.items():
        vals = [fn(r) for r in rows]
        scores[name] = {
            "auroc": auroc_from_scores(vals, labels),
            "auroc_ci95": bootstrap_auroc_ci(vals, labels, B=B, seed=seed),
            "score_min": round(min(vals), 4),
            "score_max": round(max(vals), 4),
            "live_vs_fixture_p95": round(max(vals) / fixture_taus[name]["tau_p95"], 4)
            if fixture_taus[name]["tau_p95"] > 0
            else None,
        }

    # M1: System A commensurate shipped gates
    m1_gates: dict[str, Callable[[dict[str, Any]], bool]] = {
        "system_a_recorded": lambda r: bool(r.get("abstain_recommended")),
        "fused_ge_shipped_0.82": lambda r: float(r["uncertainty_signals"]["fused"])
        >= FIXTURE_FUSED_TAU,
        "probe_ge_shipped_0.78": lambda r: float(r["affect"]["uncertainty"])
        >= FIXTURE_PROBE_TAU,
        "or_fused_probe_shipped": lambda r: (
            float(r["uncertainty_signals"]["fused"]) >= FIXTURE_FUSED_TAU
            or float(r["affect"]["uncertainty"]) >= FIXTURE_PROBE_TAU
        ),
    }
    m1_rank = {
        "system_a_recorded": "system_a_composite",
        "fused_ge_shipped_0.82": "fused_channel",
        "probe_ge_shipped_0.78": "probe_uncertainty",
        "or_fused_probe_shipped": "system_a_composite",
    }

    def _gate_block(
        gate_map: dict[str, Callable[[dict[str, Any]], bool]],
        rank_map: dict[str, str],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for gname, gfn in gate_map.items():
            n_abs = int(sum(1 for r in rows if gfn(r)))
            rank_key = rank_map[gname]
            auroc = scores[rank_key]["auroc"]
            never = n_abs == 0
            out[gname] = {
                "abstentions": f"{n_abs}/{len(rows)}",
                "n_abstain": n_abs,
                "never_abstain": never,
                "rank_score": rank_key,
                "rank_auroc": auroc,
                "silent_non_refusal": never and auroc <= 0.5,
            }
        return out

    m1 = _gate_block(m1_gates, m1_rank)

    # M2: classical (+ composite) under fixture p95 τ per channel
    m2_gates: dict[str, Callable[[dict[str, Any]], bool]] = {}
    m2_rank: dict[str, str] = {}
    for name, fn in extractors.items():
        tau = fixture_taus[name]["tau_p95"]
        gname = f"{name}_ge_fixture_p95_{tau}"
        m2_gates[gname] = (lambda r, fn=fn, tau=tau: fn(r) >= tau)
        m2_rank[gname] = name
    m2 = _gate_block(m2_gates, m2_rank)

    return {
        "n": len(rows),
        "n_pos": int(sum(labels)),
        "n_neg": int(len(labels) - sum(labels)),
        "fixture_taus": fixture_taus,
        "scores": scores,
        "m1_system_a_shipped_gates": m1,
        "m2_fixture_p95_calibrated_gates": m2,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPORTS / "protocol_m_multi_recipe_contract.json")
    args = ap.parse_args()

    models: dict[str, Any] = {}
    silent_m1: list[dict[str, Any]] = []
    silent_m2: list[dict[str, Any]] = []
    never_m1: list[dict[str, Any]] = []
    never_m2: list[dict[str, Any]] = []

    for slug, suite, label_key, live_name, fix_name in LOCK_MATRIX:
        live_path = REPORTS / live_name
        fix_path = FIXTURES / fix_name
        if not live_path.exists() or not fix_path.exists():
            models.setdefault(slug, {})[suite] = {
                "status": "missing",
                "live": str(live_path),
                "fixture": str(fix_path),
            }
            continue
        rows = _rows(live_path)
        ft = _fixture_taus(fix_path)
        block = {
            "status": "ok",
            "source_live": live_name,
            "source_fixture": fix_name,
            "label_key": label_key,
            **_eval_block(rows, label_key, ft, B=args.B, seed=args.seed),
        }
        models.setdefault(slug, {})[suite] = block
        for layer, bucket_silent, bucket_never in (
            ("m1_system_a_shipped_gates", silent_m1, never_m1),
            ("m2_fixture_p95_calibrated_gates", silent_m2, never_m2),
        ):
            for gname, g in block[layer].items():
                rec = {
                    "model": slug,
                    "suite": suite,
                    "layer": layer,
                    "gate": gname,
                    "abstentions": g["abstentions"],
                    "rank_auroc": g["rank_auroc"],
                }
                if g["never_abstain"]:
                    bucket_never.append(rec)
                if g["silent_non_refusal"]:
                    bucket_silent.append(rec)

    out = {
        "schema": "anima.protocol_m.multi_recipe_contract.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "honesty": {
            "no_new_forward_passes": True,
            "reuses_locked_claim_encode_rows": True,
            "v1_issue": (
                "v1 applied fused τ=0.82 to entropy/inv-logit-gap (incommensurate). "
                "v2 splits M1 shipped System A gates from M2 fixture-p95 per-channel τ."
            ),
            "m1": "System A commensurate shipped gates only",
            "m2": "Per-channel τ = 95th percentile on fixture sample for that channel",
        },
        "rollup": {
            "m1_n_never_abstain": len(never_m1),
            "m1_n_silent_non_refusal": len(silent_m1),
            "m2_n_never_abstain": len(never_m2),
            "m2_n_silent_non_refusal": len(silent_m2),
            "m1_silent_hits": silent_m1,
            "m2_silent_hits": silent_m2,
            "m1_never_abstain_hits": never_m1,
            "m2_never_abstain_hits": never_m2,
        },
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        f"M1 never={len(never_m1)} silent={len(silent_m1)} | "
        f"M2 never={len(never_m2)} silent={len(silent_m2)}"
    )


if __name__ == "__main__":
    main()
