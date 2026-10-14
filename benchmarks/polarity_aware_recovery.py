"""
Polarity-aware recovery (held-out only) — explicit adaptation, never silent.

Fold A (questions): diagnose anti-alignment via AUROC(composite).
Fold B: apply named policy:
  - baseline_fixture: default guard
  - disable_fused: ablation-style
  - invert_composite_abstain: research invert mode

Honesty: reports ranking AUROC + misleading-green on held-out fold only.
Does NOT change production label wiring or default guard_config.yaml.
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
    polarity_aware_policy,
)
from core.guard import evaluate_guard
from scripts.protocol_r_retune import question_holdout_split

REPORTS = Path(__file__).resolve().parent / "reports"


def _scores_labels(rows: list[dict[str, Any]], label_key: str, *, invert: bool, disable_fused: bool):
    disabled = ("fused",) if disable_fused else ()
    scores, labels = [], []
    for r in rows:
        g = evaluate_guard(
            affect=r["affect"],
            uncertainty_signals=r["uncertainty_signals"],
            token_text=r.get("token_text", ""),
            disabled_signals=disabled,
        )
        s = float(g.composite_score)
        if invert:
            s = 1.0 - s
        scores.append(s)
        labels.append(int(bool(r[label_key])))
    return scores, labels


def run_recovery(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    seed: int = 42,
) -> dict[str, Any]:
    questions = [str(r.get("question") or "") for r in rows]
    tune_m, test_m = question_holdout_split(questions, seed=seed)
    tune = [r for r, m in zip(rows, tune_m) if m]
    test = [r for r, m in zip(rows, test_m) if m]

    # Diagnose on tune fold
    s0, y0 = _scores_labels(tune, label_key, invert=False, disable_fused=False)
    auroc_tune = auroc_from_scores(s0, y0)
    anti = auroc_tune < 0.5
    # Choose adaptation from diagnosis (explicit)
    if anti:
        chosen = "invert_composite_abstain"
        invert, disable_fused = True, False
    else:
        # If not anti-aligned but fused may still hurt — try disable_fused if probe helps
        s_p, y_p = _scores_labels(tune, label_key, invert=False, disable_fused=True)
        auroc_nofused = auroc_from_scores(s_p, y_p)
        if auroc_nofused > auroc_tune + 0.02:
            chosen = "disable_fused"
            invert, disable_fused = False, True
        else:
            chosen = "baseline_fixture"
            invert, disable_fused = False, False

    policies = {
        "baseline_fixture": (False, False, default_fixture_policy),
        "disable_fused": (False, True, polarity_aware_policy(invert_composite=False, disable_fused=True)),
        "invert_composite_abstain": (
            True,
            False,
            polarity_aware_policy(invert_composite=True, disable_fused=False),
        ),
    }

    results: dict[str, Any] = {
        "diagnose_fold": {
            "n": len(tune),
            "auroc_composite": auroc_tune,
            "anti_aligned": anti,
            "chosen_policy": chosen,
        },
        "held_out_fold": {"n": len(test), "policies": {}},
        "honesty": {
            "silent_label_flip": False,
            "production_guard_config_unchanged": True,
            "adaptation_named_in_receipt": True,
            "note": "Held-out only; invert mode is research operating point, not default API.",
        },
    }

    for name, (inv, no_f, decide) in policies.items():
        scores, labels = _scores_labels(test, label_key, invert=inv, disable_fused=no_f)
        mg = misleading_green_summary(test, label_key=label_key, policy_name=name, decide=decide)
        results["held_out_fold"]["policies"][name] = {
            "auroc_composite": auroc_from_scores(scores, labels),
            "selected": name == chosen,
            "misleading_green": mg,
        }

    results["primary_held_out"] = results["held_out_fold"]["policies"][chosen]
    return results


def run_for_slug(slug: str, reports: Path = REPORTS, seed: int = 42) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_slug": slug,
        "seed": seed,
        "protocol": "polarity_aware_recovery_R_prime",
        "suites": {},
    }
    for suite, key in (("halueval", "is_hallucination"), ("truthfulqa", "should_abstain")):
        path = reports / f"{suite}_live_{slug}.json"
        if not path.is_file():
            out["suites"][suite] = {"status": "missing", "path": str(path)}
            continue
        rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
        block = run_recovery(rows, label_key=key, seed=seed)
        block["status"] = "ok"
        block["source"] = str(path)
        out["suites"][suite] = block
    return out


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Polarity-aware held-out recovery (explicit)")
    p.add_argument("--slug", required=True)
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    report = run_for_slug(args.slug, args.reports_dir, seed=args.seed)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    dest = args.reports_dir / f"polarity_aware_recovery_{args.slug}.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {dest}")
    for suite, block in report["suites"].items():
        if block.get("status") != "ok":
            print(f"  {suite}: {block.get('status')}")
            continue
        d = block["diagnose_fold"]
        prim = block["primary_held_out"]
        print(
            f"  {suite}: diagnose_auroc={d['auroc_composite']} chosen={d['chosen_policy']} "
            f"heldout_auroc={prim['auroc_composite']} "
            f"mgr={prim['misleading_green']['misleading_green_rate']}"
        )


if __name__ == "__main__":
    main()
