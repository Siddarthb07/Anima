"""Generate a publication-ready benchmark review from rollup JSON.

Uses the internal benchmark validation rubric (benchmarks.council.score_manifest).
Output: docs/BENCHMARK_PUBLISH_REVIEW.md + benchmarks/reports/validation_rollup.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.council import score_manifest  # noqa: E402
from core.layer_config import LAYER_CONFIG  # noqa: E402

REPORTS = ROOT / "benchmarks" / "reports"
OUT_MD = ROOT / "docs" / "BENCHMARK_PUBLISH_REVIEW.md"
OUT_JSON = REPORTS / "validation_rollup.json"

RUBRIC_LABELS = {
    "schema_integrity": "Manifest integrity",
    "probe_signal": "Probe signal strength",
    "honesty_flags": "Reporting honesty",
    "prompt_separation": "Live prompt separation",
}

STATUS_PHRASES = {
    "requires_gpu": "Deferred to GPU evaluation tier (not run on this hardware profile).",
    "gated_hf_repo": "Requires Hugging Face access approval; benchmark not executed.",
    "gated_hf": "Requires Hugging Face access approval; benchmark not executed.",
    "not_run_this_session": "Not evaluated in this benchmark session.",
    "error": "Benchmark execution did not complete successfully.",
    "timeout": "Benchmark exceeded allotted runtime.",
}


def _status_note(entry: dict) -> str:
    if entry.get("manifest"):
        return "Evaluated"
    reason = entry.get("reason", "")
    return STATUS_PHRASES.get(reason, reason or "Not evaluated on this hardware profile.")


def _rubric_row(verdicts: list[dict]) -> str:
    parts = []
    for v in verdicts:
        label = RUBRIC_LABELS.get(v["judge"], v["judge"])
        parts.append(f"{label}: {v['score']}/100")
    return " · ".join(parts)


def build_review(rollup: dict, validations: list[dict]) -> str:
    ts = rollup.get("generated_at", datetime.now(timezone.utc).isoformat())
    by_model = {v["model"]: v for v in validations}
    lines = [
        "# Anima benchmark — publication review",
        "",
        f"*Generated {ts} · validation rubric applied*",
        "",
        "## Purpose",
        "",
        "This document summarizes benchmark results intended for README tables, release notes,",
        "and external readers. Readouts are **instrumentation**, not claims of subjective experience.",
        "Guard metrics on synthetic fixtures measure **abstention policy behaviour**, not",
        "production hallucination detection performance.",
        "",
        "## Executive summary",
        "",
        "| Model | Validation score | Meets bar | Text probe | Evaluation status |",
        "|-------|------------------|-----------|------------|-------------------|",
    ]

    for entry in rollup.get("entries", []):
        model = entry["model"]
        v = by_model.get(model, {})
        score = v.get("aggregate_score", "—")
        meets = "yes" if v.get("passed") else "no" if v else "—"
        lines.append(
            f"| `{model}` | {score} | {meets} | "
            f"{'yes' if entry.get('has_text_probe') else 'no'} | {_status_note(entry)} |"
        )

    lines.extend(
        [
            "",
            "## Validation rubric (four dimensions)",
            "",
            "Scores are weighted 0–100 per dimension; aggregate ≥60 with core dimensions passing",
            "indicates the model is suitable to cite in portfolio materials with stated limits.",
            "",
            "| Dimension | Weight | What it checks |",
            "|-----------|--------|----------------|",
            "| Manifest integrity | 15% | Schema, timestamps, complete benchmark entries |",
            "| Probe signal strength | 35% | GoEmotions Pearson r, brain holdout r, smoke extract |",
            "| Reporting honesty | 20% | Flags overclaim patterns (e.g. perfect AUROC on small fixtures) |",
            "| Live prompt separation | 30% | Positive vs negative prompt mean-valence gap |",
            "",
        ]
    )

    for entry in rollup.get("entries", []):
        model = entry["model"]
        v = by_model.get(model, {})
        lines.extend([f"## `{model}`", ""])
        if not entry.get("manifest"):
            lines.append(f"*{_status_note(entry)}*")
            lines.extend(["", "---", ""])
            continue

        lines.append(
            f"**Validation score:** {v.get('aggregate_score', '—')}/100 · "
            f"**Meets publication bar:** {'yes' if v.get('passed') else 'no'}"
        )
        if v.get("verdicts"):
            lines.append(f"**Rubric:** {_rubric_row(v['verdicts'])}")
            lines.extend(["", "**Reviewer notes:**", ""])
            for dim in v["verdicts"]:
                label = RUBRIC_LABELS.get(dim["judge"], dim["judge"])
                notes = "; ".join(dim.get("notes") or []) or "No issues flagged."
                lines.append(f"- *{label}:* {notes}")

        go = next(
            (e for e in (entry.get("manifest") or {}).get("entries", []) if e.get("benchmark") == "go_emotions"),
            None,
        )
        if go and go.get("status") == "ok":
            lines.extend(
                [
                    "",
                    f"- GoEmotions validation Pearson r (valence): **{go.get('pearson_valence')}**",
                ]
            )

        good = v.get("good_examples") or []
        weak = v.get("bad_examples") or []
        if good:
            lines.extend(["", "**Supporting examples:**", ""])
            for ex in good:
                lines.append(
                    f"- `{ex.get('id')}`: mean valence {ex.get('mean_valence')} — {ex.get('verdict', '')}"
                )
        if weak:
            lines.extend(["", "**Known limits (state explicitly in README):**", ""])
            for ex in weak:
                lines.append(
                    f"- `{ex.get('id')}`: mean valence {ex.get('mean_valence')} — {ex.get('verdict', '')}"
                )
        lines.extend(["", "---", ""])

    lines.extend(
        [
            "## Recommended README claims",
            "",
            "Only cite metrics for models that **Meets bar = yes** and where the supporting",
            "example exists. Always pair numbers with the evaluation tier (CPU, synthetic Narratives,",
            "fixture guard policy).",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "$env:ANIMA_FORCE_CPU='1'",
            "$env:NARRATIVES_ROOT='.\\data\\narratives_minimal'",
            "python scripts/run_all_models_benchmark.py",
            "python scripts/generate_benchmark_report.py",
            "python scripts/generate_benchmark_charts.py",
            "python scripts/benchmark_publish_review.py",
            "```",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate publication-ready benchmark review")
    p.add_argument("--rollup", default=str(REPORTS / "all_models_rollup.json"))
    args = p.parse_args()

    rollup_path = Path(args.rollup)
    if not rollup_path.exists():
        print(f"Missing rollup: {rollup_path}", file=sys.stderr)
        return 1

    rollup = json.loads(rollup_path.read_text(encoding="utf-8"))
    run_models = {e["model"] for e in rollup.get("entries", [])}
    for model_id, cfg in LAYER_CONFIG.items():
        if model_id in run_models:
            continue
        reason = "requires_gpu" if cfg.get("requires_gpu") else "not_run_this_session"
        rollup.setdefault("entries", []).append(
            {
                "model": model_id,
                "status": "deferred",
                "reason": reason,
                "manifest": None,
                "prompt_examples": [],
                "has_text_probe": False,
                "has_brain_probe": False,
            }
        )

    validations: list[dict] = []
    for entry in rollup.get("entries", []):
        manifest = entry.get("manifest")
        if not manifest or not isinstance(manifest, dict):
            continue
        report = score_manifest(manifest, entry.get("prompt_examples") or [])
        row = report.to_dict()
        row["evaluation_status"] = "completed"
        validations.append(row)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(validations, indent=2), encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_review(rollup, validations), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
