"""
Compute Trust Integrity Gate receipts from locked Protocol F/L or Protocol L′.

Usage:
  python -m benchmarks.run_trust_integrity_gate --all-locked
  python -m benchmarks.run_trust_integrity_gate --all-locked --adapted
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from core.trust_integrity import (
    ADAPTED_CRITERIA,
    SuiteIntegrity,
    evaluate_trust_integrity,
    transfer_gap,
    channel_sign_conflict,
    write_receipt,
)

REPORTS = Path(__file__).resolve().parent / "reports"


def _load(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_auroc(suite: str) -> Optional[float]:
    from benchmarks.guard_metrics import load_fixture, score_abstain_accuracy

    fname = "halueval_guard_sample.json" if suite == "halueval" else "truthfulqa_guard_sample.json"
    key = "is_hallucination" if suite == "halueval" else "should_abstain"
    rows = load_fixture(Path(__file__).resolve().parent / "fixtures" / fname)
    return float(score_abstain_accuracy(rows, label_key=key)["auroc_composite"])


def build_suites(slug: str, reports: Path = REPORTS) -> tuple[dict[str, SuiteIntegrity], list[str]]:
    sources: list[str] = []
    live = _load(reports / f"live_guard_{slug}.json") or {}
    pol = _load(reports / f"polarity_diagnostic_{slug}.json") or {}
    ab = _load(reports / f"guard_ablation_{slug}.json") or {}
    suites: dict[str, SuiteIntegrity] = {}

    for suite in ("halueval", "truthfulqa"):
        bench = (live.get("benchmarks") or {}).get(suite) or {}
        rows_path = reports / f"{suite}_live_{slug}.json"
        row_doc = _load(rows_path) or {}
        metrics = bench or (row_doc.get("metrics") or {})
        if metrics:
            sources.append(str(rows_path if rows_path.is_file() else f"live_guard_{slug}.json:{suite}"))

        auroc_l = metrics.get("auroc_composite")
        auroc_f = _fixture_auroc(suite)
        gap = transfer_gap(auroc_f, auroc_l) if auroc_f is not None and auroc_l is not None else None

        pol_suite = (pol.get("suites") or {}).get(suite) or {}
        abl = ((ab.get(suite) or {}).get("ablations") or {})
        probe = (abl.get("probe_only") or {}).get("auroc_composite")
        fused = (abl.get("fused_only") or {}).get("auroc_composite")

        suites[suite] = SuiteIntegrity(
            suite=suite,
            protocol_f_auroc=auroc_f,
            protocol_l_auroc=float(auroc_l) if auroc_l is not None else None,
            transfer_gap=gap,
            auroc_ci95=metrics.get("auroc_ci95"),
            never_abstains=bool(metrics.get("never_abstains"))
            if "never_abstains" in metrics
            else (metrics.get("tp", 1) == 0 and metrics.get("fp", 1) == 0),
            n_live=metrics.get("n_samples"),
            polarity_verdict=pol_suite.get("verdict"),
            probe_only_auroc=float(probe) if probe is not None else None,
            fused_only_auroc=float(fused) if fused is not None else None,
            channel_sign_conflict=channel_sign_conflict(
                float(probe) if probe is not None else None,
                float(fused) if fused is not None else None,
            ),
        )
        if pol_suite:
            sources.append(f"polarity_diagnostic_{slug}.json:{suite}")
        if abl:
            sources.append(f"guard_ablation_{slug}.json:{suite}")

    return suites, sources


def build_suites_l_prime(
    slug: str, reports: Path = REPORTS
) -> tuple[dict[str, SuiteIntegrity], list[str]]:
    """Build suites from Protocol L′ LOQO report (run protocol_l_prime first)."""
    doc = _load(reports / f"protocol_l_prime_{slug}.json")
    if not doc:
        raise FileNotFoundError(
            f"Missing protocol_l_prime_{slug}.json — run: "
            f"python -m benchmarks.protocol_l_prime --slug {slug}"
        )
    sources = [str(reports / f"protocol_l_prime_{slug}.json")]
    suites: dict[str, SuiteIntegrity] = {}
    for suite, block in (doc.get("suites") or {}).items():
        if block.get("status") != "ok":
            continue
        mgr = (block.get("misleading_green") or {}).get("misleading_green_rate")
        suites[suite] = SuiteIntegrity(
            suite=suite,
            protocol_f_auroc=block.get("protocol_f_same_scorer_auroc"),
            protocol_l_auroc=block.get("auroc_loqo"),
            transfer_gap=block.get("transfer_gap_same_scorer"),
            auroc_ci95=block.get("auroc_ci95"),
            never_abstains=bool(block.get("never_abstains")),
            n_live=block.get("n"),
            polarity_verdict=block.get("polarity_verdict"),
            probe_only_auroc=None,
            fused_only_auroc=None,
            channel_sign_conflict=False,
            misleading_green_rate=float(mgr) if mgr is not None else None,
        )
    return suites, sources


def run_for_slug(
    *,
    slug: str,
    model: str,
    reports: Path = REPORTS,
    adapted: bool = False,
) -> Any:
    if adapted:
        suites, sources = build_suites_l_prime(slug, reports)
        return evaluate_trust_integrity(
            model=model,
            model_slug=slug,
            suites=suites,
            criteria=ADAPTED_CRITERIA,
            sources=sources,
        )
    suites, sources = build_suites(slug, reports)
    return evaluate_trust_integrity(
        model=model,
        model_slug=slug,
        suites=suites,
        sources=sources,
    )


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Trust Integrity Gate — green-shield claim lock")
    p.add_argument("--slug", default="", help="Model slug (required unless --all-locked)")
    p.add_argument("--model", default="")
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    p.add_argument("--all-locked", action="store_true")
    p.add_argument(
        "--adapted",
        action="store_true",
        help="Score Protocol L′ LOQO operating point (writes trust_integrity_adapted_*.json)",
    )
    args = p.parse_args(argv)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.all_locked:
        jobs = [
            ("qwen_qwen2_5-0_5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
            ("tinyllama_tinyllama-1_1b-chat-v1_0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        ]
    else:
        if not args.slug:
            p.error("--slug is required unless --all-locked")
        jobs = [(args.slug, args.model or args.slug)]

    rollup: dict[str, Any] = {
        "status": "ok",
        "operating_point": "protocol_L_prime_loqo" if args.adapted else "protocol_L_baseline",
        "receipts": {},
    }
    for slug, model in jobs:
        receipt = run_for_slug(
            slug=slug, model=model, reports=args.reports_dir, adapted=args.adapted
        )
        suffix = f"adapted_{slug}" if args.adapted else slug
        out = args.reports_dir / f"trust_integrity_{suffix}.json"
        write_receipt(receipt, out)
        rollup["receipts"][slug] = {
            "path": str(out),
            "trust_ui_claim_allowed": receipt.trust_ui_claim_allowed,
            "failed_gates": receipt.failed_gates,
            "passed_gates": receipt.passed_gates,
            "transfer_gap_max": receipt.transfer_gap_max,
            "seal_sha256": receipt.seal_sha256,
            "operating_point": receipt.operating_point,
        }
        print(
            f"[{slug}] op={receipt.operating_point} claim_allowed={receipt.trust_ui_claim_allowed} "
            f"failed={len(receipt.failed_gates)} gap_max={receipt.transfer_gap_max}"
        )
        for g in receipt.failed_gates:
            print(f"  FAIL {g}")
        print(f"  Wrote {out}")

    rollup["any_trust_ui_claim_allowed"] = any(
        r["trust_ui_claim_allowed"] for r in rollup["receipts"].values()
    )
    rollup["innovation"] = (
        "Trust Integrity Gate: fixture-perfect green shields stay locked until "
        "live criteria pass (baseline L or adapted L′)."
    )
    name = "trust_integrity_adapted_rollup.json" if args.adapted else "trust_integrity_rollup.json"
    rollup_path = args.reports_dir / name
    rollup_path.write_text(json.dumps(rollup, indent=2), encoding="utf-8")
    print(f"Wrote {rollup_path}")


if __name__ == "__main__":
    main()
