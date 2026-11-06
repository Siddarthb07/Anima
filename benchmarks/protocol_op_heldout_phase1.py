#!/usr/bin/env python3
"""Phase-1 held-out operating point seal (CPU; no new LM forwards).

Pre-registered policy name: invert_composite_abstain (chosen on tune fold when
AUROC < 0.5; else disable_fused / baseline_fixture via polarity_aware_recovery).

Emits:
  - abstain k/n on held-out fold
  - ranking AUROC
  - misleading_green_rate (= false-green rate under green-on-non-abstain)
  - coverage_at_risk_0.10 / coverage_at_risk_0.20 (selective prediction)
  - shipping_claim=false

Does NOT mutate probes/guard_config.yaml.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from benchmarks.guard_metrics import auroc_from_scores
from benchmarks.misleading_green import (
    default_fixture_policy,
    misleading_green_summary,
)
from benchmarks.polarity_aware_recovery import run_recovery, _scores_labels
from benchmarks.risk_coverage_and_length_match import risk_coverage
from scripts.protocol_r_retune import question_holdout_split

REPORTS = Path(__file__).resolve().parent / "reports"
PAPER_ART = Path(
    r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact"
)

PROTOCOL = "protocol_op_heldout_phase1.v1"
PRE_REGISTERED = {
    "seed": 42,
    "anti_aligned_policy": "invert_composite_abstain",
    "invert_threshold": 0.55,
    "probe_threshold": 0.78,
    "risk_targets": [0.10, 0.20],
    "shipping_claim": False,
}


def _coverage_at_risk(
    scores: list[float],
    labels: list[int],
    *,
    risk_target: float,
    higher_is_risk: bool = True,
) -> dict[str, Any]:
    """Largest coverage such that positive_rate_on_answered <= risk_target."""
    curve = risk_coverage(scores, labels, higher_is_risk=higher_is_risk)
    feasible = [
        p
        for p in curve
        if p.get("positive_rate_on_answered") is not None
        and float(p["positive_rate_on_answered"]) <= risk_target + 1e-9
    ]
    if not feasible:
        return {
            "risk_target": risk_target,
            "coverage": 0.0,
            "abstain_frac": 1.0,
            "positive_rate_on_answered": None,
            "status": "infeasible",
        }
    best = max(feasible, key=lambda p: float(p["coverage"]))
    return {
        "risk_target": risk_target,
        "coverage": best["coverage"],
        "abstain_frac": best["abstain_frac"],
        "positive_rate_on_answered": best["positive_rate_on_answered"],
        "status": "ok",
    }


def _enrich_suite(rows: list[dict[str, Any]], label_key: str, seed: int) -> dict[str, Any]:
    base = run_recovery(rows, label_key=label_key, seed=seed)
    questions = [str(r.get("question") or "") for r in rows]
    tune_m, test_m = question_holdout_split(questions, seed=seed)
    test = [r for r, m in zip(rows, test_m) if m]
    chosen = base["diagnose_fold"]["chosen_policy"]
    invert = chosen == "invert_composite_abstain"
    disable_fused = chosen == "disable_fused"
    scores, labels = _scores_labels(test, label_key, invert=invert, disable_fused=disable_fused)
    # For inverted composite, higher inverted score = higher risk → higher_is_risk True
    cov = {
        f"coverage_at_risk_{rt:.2f}": _coverage_at_risk(
            scores, labels, risk_target=rt, higher_is_risk=True
        )
        for rt in PRE_REGISTERED["risk_targets"]
    }
    # Baseline fixture on same held-out fold for delta
    base_scores, base_labels = _scores_labels(
        test, label_key, invert=False, disable_fused=False
    )
    base_mg = misleading_green_summary(
        test, label_key=label_key, policy_name="baseline_fixture", decide=default_fixture_policy
    )
    chosen_mg = base["primary_held_out"]["misleading_green"]
    return {
        **base,
        "reliability_quantities": {
            "held_out_n": len(test),
            "chosen_policy": chosen,
            "held_out_auroc": auroc_from_scores(scores, labels),
            "abstain_events": f"{chosen_mg['n_abstain']}/{chosen_mg['n']}",
            "abstain_rate": chosen_mg["abstain_rate"],
            "false_green_rate": chosen_mg["misleading_green_rate"],
            "error_among_green": chosen_mg["error_among_green"],
            "baseline_fixture_abstain_events": f"{base_mg['n_abstain']}/{base_mg['n']}",
            "baseline_fixture_false_green_rate": base_mg["misleading_green_rate"],
            "coverage_at_risk": cov,
            "baseline_coverage_at_risk_0.10": _coverage_at_risk(
                base_scores, base_labels, risk_target=0.10, higher_is_risk=True
            ),
        },
    }


def run_slug(slug: str, reports: Path, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": PROTOCOL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_slug": slug,
        "seed": seed,
        "pre_registered": PRE_REGISTERED,
        "shipping_claim": False,
        "suites": {},
    }
    for suite, key in (("halueval", "is_hallucination"), ("truthfulqa", "should_abstain")):
        path = reports / f"{suite}_live_{slug}.json"
        if not path.is_file():
            out["suites"][suite] = {"status": "missing", "path": str(path)}
            continue
        rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
        block = _enrich_suite(rows, key, seed)
        block["status"] = "ok"
        block["source"] = str(path)
        out["suites"][suite] = block
    return out


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Phase-1 held-out OP seal")
    p.add_argument(
        "--slug",
        action="append",
        dest="slugs",
        default=None,
        help="Repeatable; default = locked System A trio",
    )
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--also-paper-artifact", action="store_true")
    args = p.parse_args(argv)
    slugs = args.slugs or [
        "qwen_qwen2_5-0_5b-instruct",
        "tinyllama_tinyllama-1_1b-chat-v1_0",
        "distilgpt2",
    ]
    rollup: dict[str, Any] = {
        "schema": PROTOCOL + ".rollup",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pre_registered": PRE_REGISTERED,
        "shipping_claim": False,
        "models": {},
        "headline": {},
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    for slug in slugs:
        report = run_slug(slug, args.reports_dir, args.seed)
        dest = args.reports_dir / f"protocol_op_heldout_phase1_{slug}.json"
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {dest}")
        rollup["models"][slug] = report
        for suite, block in report["suites"].items():
            if block.get("status") != "ok":
                continue
            rq = block["reliability_quantities"]
            print(
                f"  {suite}: policy={rq['chosen_policy']} abs={rq['abstain_events']} "
                f"auroc={rq['held_out_auroc']:.4f} false_green={rq['false_green_rate']} "
                f"cov@0.10={rq['coverage_at_risk']['coverage_at_risk_0.10']}"
            )

    # Headline cell: Qwen HaluEval invert OP
    q = rollup["models"].get("qwen_qwen2_5-0_5b-instruct", {}).get("suites", {}).get("halueval")
    if q and q.get("status") == "ok":
        rq = q["reliability_quantities"]
        rollup["headline"] = {
            "cell": "qwen_0.5b_halueval_heldout",
            "policy": rq["chosen_policy"],
            "abstain_events": rq["abstain_events"],
            "held_out_auroc": rq["held_out_auroc"],
            "false_green_rate": rq["false_green_rate"],
            "error_among_green": rq["error_among_green"],
            "coverage_at_risk_0.10": rq["coverage_at_risk"]["coverage_at_risk_0.10"],
            "coverage_at_risk_0.20": rq["coverage_at_risk"]["coverage_at_risk_0.20"],
            "baseline_never_abstains": rq["baseline_fixture_abstain_events"] == "0/40"
            or rq["baseline_fixture_abstain_events"].startswith("0/"),
            "shipping_claim": False,
            "note": (
                "Held-out question fold only; invert_composite_abstain is a "
                "research OP, not production guard_config.yaml."
            ),
        }

    rollup_path = args.reports_dir / "protocol_op_heldout_phase1_rollup.json"
    rollup_path.write_text(json.dumps(rollup, indent=2), encoding="utf-8")
    print(f"Wrote {rollup_path}")

    if args.also_paper_artifact:
        PAPER_ART.mkdir(parents=True, exist_ok=True)
        for slug in slugs:
            src = args.reports_dir / f"protocol_op_heldout_phase1_{slug}.json"
            if src.is_file():
                (PAPER_ART / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        (PAPER_ART / "protocol_op_heldout_phase1_rollup.json").write_text(
            json.dumps(rollup, indent=2), encoding="utf-8"
        )
        print(f"Copied seals → {PAPER_ART}")


if __name__ == "__main__":
    main()
