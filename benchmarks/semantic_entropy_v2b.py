#!/usr/bin/env python3
"""SE v2-B: per-row answer-span uncertainty from locked live encode rows.

Additive only. Does NOT overwrite semantic_entropy_matched_*.json or live_guard_*.json.

Source rows:
  benchmarks/reports/{halueval,truthfulqa}_live_<slug>.json
  (uncertainty_signals already answer-span averaged in live_guard_eval)

Also seals claim_disagreement AUROC from matched SE receipts as the
per-row SE-family control (v1 raw SE remains the degeneracy exhibit).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.guard_metrics import auroc_from_scores, bootstrap_auroc_ci

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "benchmarks" / "reports"

SLUGS = (
    "qwen_qwen2_5-0_5b-instruct",
    "tinyllama_tinyllama-1_1b-chat-v1_0",
)
SUITES = ("halueval", "truthfulqa")


def _pack(scores: list[float], labels: list[int], *, B: int = 2000, seed: int = 42) -> dict[str, Any]:
    au = float(auroc_from_scores(scores, labels))
    ci = bootstrap_auroc_ci(scores, labels, B=B, seed=seed)
    return {"auroc": round(au, 4), "auroc_ci95": ci}


def _load_live_rows(suite: str, slug: str) -> list[dict[str, Any]]:
    p = REPORTS / f"{suite}_live_{slug}.json"
    if not p.is_file():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    return list(d.get("rows") or [])


def _label_key(suite: str) -> str:
    return "is_hallucination" if suite == "halueval" else "should_abstain"


def score_suite(suite: str, slug: str, *, B: int = 2000, seed: int = 42) -> dict[str, Any] | None:
    rows = _load_live_rows(suite, slug)
    if not rows:
        return None
    lk = _label_key(suite)
    labels = [int(bool(r[lk])) for r in rows]
    ent = [float(r["uncertainty_signals"]["entropy"]) for r in rows]
    fused = [float(r["uncertainty_signals"]["fused"]) for r in rows]
    # Higher logit gap = more confident; invert so higher score = more uncertain.
    gap_unc = [1.0 - float(r["uncertainty_signals"]["logit_gap"]) for r in rows]

    controls: dict[str, Any] = {
        "answer_span_entropy": _pack(ent, labels, B=B, seed=seed),
        "answer_span_fused": _pack(fused, labels, B=B, seed=seed),
        "answer_span_inv_logit_gap": _pack(gap_unc, labels, B=B, seed=seed),
    }

    se_path = REPORTS / f"semantic_entropy_matched_{slug}.json"
    if se_path.is_file():
        se = json.loads(se_path.read_text(encoding="utf-8"))
        bas = ((se.get("suites") or {}).get(suite) or {}).get("baselines") or {}
        controls["semantic_entropy_v1"] = bas.get("semantic_entropy")
        controls["claim_disagreement"] = bas.get("claim_disagreement")
        controls["anima_composite_ref"] = bas.get("anima_composite")
        controls["probe_only_ref"] = bas.get("probe_only")

    return {
        "n": len(rows),
        "label_key": lk,
        "controls": controls,
        "ieee_note": (
            "answer_span_* are per-row scores from locked claim-encode uncertainty_signals; "
            "not Farquhar multi-sample SE. v1 SE stays the degeneracy exhibit."
        ),
    }


def build_report(*, B: int = 2000, seed: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for slug in SLUGS:
        suites: dict[str, Any] = {}
        model_name = None
        for suite in SUITES:
            scored = score_suite(suite, slug, B=B, seed=seed)
            if scored is None:
                continue
            live_meta = REPORTS / f"{suite}_live_{slug}.json"
            if live_meta.is_file() and model_name is None:
                model_name = json.loads(live_meta.read_text(encoding="utf-8")).get("model")
            suites[suite] = scored
        if suites:
            models[slug] = {"model": model_name, "model_slug": slug, "suites": suites}

    return {
        "schema": "anima.semantic_entropy_v2b.answer_span",
        "status": "ok",
        "protocol": "L_answer_span_uncertainty_reanalysis",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "B": B,
        "seed": seed,
        "honesty": {
            "baseline_lock": True,
            "v1_unchanged": True,
            "source": "{suite}_live_<slug>.json uncertainty_signals (answer-span mean)",
            "not_farquhar_multisample_se": True,
            "baseline_files_untouched": [
                "semantic_entropy_matched_*.json",
                "live_guard_*.json",
                "trust_integrity_*.json",
            ],
        },
        "models": models,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPORTS / "semantic_entropy_v2b_answer_span.json",
    )
    args = ap.parse_args()
    report = build_report(B=args.B, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    for slug, m in (report.get("models") or {}).items():
        for suite, s in (m.get("suites") or {}).items():
            c = s["controls"]
            print(
                f"  {slug} {suite}: "
                f"ent={c['answer_span_entropy']['auroc']} "
                f"fused={c['answer_span_fused']['auroc']} "
                f"inv_gap={c['answer_span_inv_logit_gap']['auroc']} "
                f"claim_dis={(c.get('claim_disagreement') or {}).get('auroc')} "
                f"se_v1={(c.get('semantic_entropy_v1') or {}).get('auroc')}"
            )


if __name__ == "__main__":
    main()
