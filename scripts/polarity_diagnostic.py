"""
A0 — Polarity diagnostic on locked live guard rows (offline).

Invert binary labels and recompute AUROC. If flipped ≈ 1 − original,
ranking is anti-aligned (paper claim OK). If flipped ≈ original, bug hunt.

Does NOT change production label wiring in live_guard_eval.load_halueval_pairs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from benchmarks.guard_metrics import auroc_from_scores, bootstrap_auroc_ci

REPORTS = Path(__file__).resolve().parent.parent / "benchmarks" / "reports"
NOTES = Path(__file__).resolve().parent.parent / "docs" / "notes"


def _label_key(rows: list[dict[str, Any]]) -> str:
    if rows and "is_hallucination" in rows[0]:
        return "is_hallucination"
    if rows and "should_abstain" in rows[0]:
        return "should_abstain"
    raise ValueError("rows missing is_hallucination / should_abstain")


def diagnose_rows(
    rows: list[dict[str, Any]],
    *,
    score_key: str = "composite_score",
    B: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    label_key = _label_key(rows)
    scores = [float(r[score_key]) for r in rows]
    y = [int(bool(r[label_key])) for r in rows]
    y_flip = [1 - v for v in y]
    auroc = auroc_from_scores(scores, y)
    auroc_flip = auroc_from_scores(scores, y_flip)
    ci = bootstrap_auroc_ci(scores, y, B=B, seed=seed)
    ci_flip = bootstrap_auroc_ci(scores, y_flip, B=B, seed=seed)
    pos = [s for s, lab in zip(scores, y) if lab == 1]
    neg = [s for s, lab in zip(scores, y) if lab == 0]
    mean_pos = sum(pos) / len(pos) if pos else float("nan")
    mean_neg = sum(neg) / len(neg) if neg else float("nan")
    expected_flip = round(1.0 - auroc, 4) if auroc == auroc else None
    delta_vs_one_minus = (
        round(abs(auroc_flip - (1.0 - auroc)), 4) if expected_flip is not None else None
    )
    # IEEE-style verdict:
    # Flip AUROC is always ≈ 1−original for a coherent ranking (identity of Mann–Whitney).
    # That identity alone does NOT mean anti-alignment — only AUROC < 0.5 does.
    if abs(auroc_flip - auroc) <= 0.05 and not (
        delta_vs_one_minus is not None and delta_vs_one_minus <= 0.01
    ):
        verdict = "bug_hunt_needed"
        note = "Flip did not invert AUROC; investigate label construction or score NaNs."
    elif delta_vs_one_minus is not None and delta_vs_one_minus <= 0.01:
        if auroc < 0.45:
            verdict = "anti_aligned_ranking"
            note = (
                f"AUROC={auroc} < 0.45 and flipped≈1−original ({expected_flip}): "
                "score ranks opposite the intended positive class; not a label-wire bug."
            )
        elif auroc < 0.55:
            verdict = "near_chance_coherent"
            note = (
                f"AUROC={auroc} near chance; flip≈1−original confirms coherent ranking "
                "(not a label bug), but not usable for trust UI."
            )
        else:
            verdict = "aligned_ranking"
            note = (
                f"AUROC={auroc} ≥ 0.55 and flip≈1−original: ranking agrees with labels "
                "(label wiring OK)."
            )
    else:
        verdict = "partial_or_noisy"
        note = (
            f"Flip moved AUROC {auroc} → {auroc_flip} but not cleanly to 1−original; "
            "inspect class means and CI overlap."
        )
    return {
        "label_key": label_key,
        "score_key": score_key,
        "n": len(rows),
        "n_pos": sum(y),
        "n_neg": len(y) - sum(y),
        "auroc_original": auroc,
        "auroc_flipped": auroc_flip,
        "expected_flip_if_anti_aligned": expected_flip,
        "abs_delta_flip_vs_one_minus_original": delta_vs_one_minus,
        "mean_score_label_1": round(mean_pos, 4),
        "mean_score_label_0": round(mean_neg, 4),
        "auroc_ci95_original": ci,
        "auroc_ci95_flipped": ci_flip,
        "verdict": verdict,
        "note": note,
        "protocol": "A0_polarity_diagnostic",
        "integrity": "production_labels_unchanged",
    }


def run_diagnostic(
    *,
    reports_dir: Path = REPORTS,
    slug: str = "qwen_qwen2_5-0_5b-instruct",
    B: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    suites = {
        "halueval": reports_dir / f"halueval_live_{slug}.json",
        "truthfulqa": reports_dir / f"truthfulqa_live_{slug}.json",
    }
    report: dict[str, Any] = {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_slug": slug,
        "seed": seed,
        "B": B,
        "suites": {},
    }
    for name, path in suites.items():
        if not path.is_file():
            report["suites"][name] = {"status": "missing", "path": str(path)}
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("rows") or []
        diag = diagnose_rows(rows, B=B, seed=seed)
        diag["source_path"] = str(path)
        diag["status"] = "ok"
        report["suites"][name] = diag
    return report


def write_note(report: dict[str, Any], dest: Path) -> None:
    lines = [
        "# A0 Polarity diagnostic",
        "",
        f"Generated: `{report.get('generated_at')}`",
        f"Model slug: `{report.get('model_slug')}`",
        "",
        "Production HaluEval label wiring is **unchanged** "
        "(correct→False / hallucinated→True). This offline invert tests "
        "whether AUROC < 0.5 is anti-aligned ranking vs a label bug.",
        "",
    ]
    for suite, block in (report.get("suites") or {}).items():
        lines.append(f"## {suite}")
        lines.append("")
        if block.get("status") != "ok":
            lines.append(f"- status: `{block.get('status')}` path=`{block.get('path')}`")
            lines.append("")
            continue
        lines.extend(
            [
                f"- n={block['n']} (pos={block['n_pos']}, neg={block['n_neg']})",
                f"- AUROC original: **{block['auroc_original']}** "
                f"CI [{block['auroc_ci95_original']['lo']}, {block['auroc_ci95_original']['hi']}]",
                f"- AUROC flipped: **{block['auroc_flipped']}** "
                f"(expected if anti-aligned: {block['expected_flip_if_anti_aligned']})",
                f"- mean score label=1: {block['mean_score_label_1']}; "
                f"label=0: {block['mean_score_label_0']}",
                f"- **verdict:** `{block['verdict']}` — {block['note']}",
                "",
            ]
        )
    lines.extend(
        [
            "## IEEE / TMLR integrity",
            "",
            "- Do not silently flip labels in `live_guard_eval` to manufacture AUROC > 0.5.",
            "- Cite live AUROC with CI; lead with collapse, not fixture 1.00.",
            "",
        ]
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="A0 polarity diagnostic (offline label invert)")
    p.add_argument("--slug", default="qwen_qwen2_5-0_5b-instruct")
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    report = run_diagnostic(reports_dir=args.reports_dir, slug=args.slug, B=args.B, seed=args.seed)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.reports_dir / f"polarity_diagnostic_{args.slug}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    note_path = NOTES / "polarity_diagnostic.md"
    write_note(report, note_path)
    print(f"Wrote {out_json}")
    print(f"Wrote {note_path}")
    for suite, block in report["suites"].items():
        if block.get("status") == "ok":
            print(
                f"  {suite}: auroc={block['auroc_original']} "
                f"flip={block['auroc_flipped']} verdict={block['verdict']}"
            )


if __name__ == "__main__":
    main()
