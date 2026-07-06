"""Run LLM council judges on rollup manifests; write council scores + BENCHMARK_REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.council import score_manifest, write_council_json  # noqa: E402
from core.layer_config import LAYER_CONFIG  # noqa: E402

REPORTS = ROOT / "benchmarks" / "reports"
DOCS = ROOT / "docs" / "BENCHMARK_REPORT.md"


def _fmt_entry(e: dict) -> str:
    b = e.get("benchmark", "?")
    st = e.get("status", "?")
    if st != "ok":
        return f"| {b} | {st} | {e.get('reason', e.get('message', '—'))} |"
    parts = []
    if "pearson_valence" in e:
        parts.append(f"r_v={e['pearson_valence']}")
    if "val_r_valence" in e:
        parts.append(f"brain_r_v={e['val_r_valence']}")
    if "abstain_accuracy" in e:
        parts.append(f"guard_acc={e['abstain_accuracy']}")
    if "n_tokens" in e:
        parts.append(f"n={e['n_tokens']}")
    return f"| {b} | ok | {', '.join(parts) or '—'} |"


def _example_block(ex: dict, good: bool) -> str:
    tag = "Good" if good else "Weak"
    lines = [
        f"**{tag} — `{ex.get('id')}`**",
        f"- Prompt: *{ex.get('prompt', ex.get('text', '—'))}*",
        f"- Mean valence: **{ex.get('mean_valence', '—')}** · arousal: {ex.get('mean_arousal', '—')}",
    ]
    if ex.get("verdict"):
        lines.append(f"- Rubric note: {ex['verdict']}")
    if ex.get("sample_output"):
        lines.append(f"- Sample output: `{ex['sample_output'][:120]}…`" if len(ex.get("sample_output", "")) > 120 else f"- Sample output: `{ex.get('sample_output')}`")
    return "\n".join(lines)


def build_markdown(rollup: dict, councils: list[dict]) -> str:
    ts = rollup.get("generated_at", datetime.now(timezone.utc).isoformat())
    lines = [
        "# Anima benchmark report",
        "",
        f"*Generated {ts} · validation rubric applied*",
        "",
        "## Executive summary",
        "",
        "This report aggregates benchmarks across all registered HuggingFace models in `core/layer_config.py`.",
        "Readouts are **instrumentation**, not claims that models feel emotions.",
        "Guard AUROC scores on synthetic fixtures are **policy smoke tests**, not hallucination detection benchmarks.",
        "",
        "| Model | Validation score | Meets bar | Text probe | Brain probe |",
        "|-------|------------------|-----------|------------|-------------|",
    ]
    council_by_model = {c["model"]: c for c in councils}
    for entry in rollup.get("entries", []):
        model = entry["model"]
        c = council_by_model.get(model, {})
        score = c.get("aggregate_score", "—")
        passed = "yes" if c.get("passed") else "no" if c else "—"
        lines.append(
            f"| `{model}` | {score} | {passed} | "
            f"{'yes' if entry.get('has_text_probe') else 'no'} | "
            f"{'yes' if entry.get('has_brain_probe') else 'no'} |"
        )

    lines.extend(["", "## Benchmark validation rubric", ""])
    lines.extend(
        [
            "Four rule-based dimensions score each model (0–100, weighted):",
            "",
            "1. **schema_integrity** (15%) — manifest schema, timestamps, entries present",
            "2. **probe_signal** (35%) — GoEmotions Pearson r, brain holdout r, smoke extract",
            "3. **honesty_flags** (20%) — penalises perfect AUROC on tiny fixtures, n<50",
            "4. **prompt_separation** (30%) — positive vs negative live prompt mean-valence gap",
            "",
            "Aggregate ≥60 with core dimensions passing = **meets publication bar**.",
            "",
        ]
    )

    for entry in rollup.get("entries", []):
        model = entry["model"]
        manifest = entry.get("manifest")
        c = council_by_model.get(model, {})
        lines.extend([f"## `{model}`", ""])
        if entry.get("status") in ("skipped", "deferred"):
            reason = entry.get("reason", "")
            phrases = {
                "requires_gpu": "Deferred to GPU evaluation tier (not evaluated on this hardware profile).",
                "gated_hf_repo": "Requires Hugging Face access approval; benchmark not executed.",
                "gated_hf": "Requires Hugging Face access approval; benchmark not executed.",
            }
            lines.append(f"*{phrases.get(reason, reason or 'Not evaluated on this hardware profile.')}*")
            lines.extend(["", "---", ""])
            continue
        if not manifest:
            lines.append("*Benchmark failed or manifest missing.*")
            lines.extend(["", "---", ""])
            continue

        lines.extend(
            [
                f"**Validation score:** {c.get('aggregate_score', '—')}/100 · "
                f"**Meets bar:** {'yes' if c.get('passed') else 'no'}",
                "",
                "### Benchmark entries",
                "",
                "| Benchmark | Status | Metrics |",
                "|-----------|--------|---------|",
            ]
        )
        for e in manifest.get("entries", []):
            lines.append(_fmt_entry(e))

        lines.extend(["", "### Rubric dimension notes", ""])
        for v in c.get("verdicts", []):
            lines.append(f"- **{v['judge']}** ({v['score']}/100): " + "; ".join(v.get("notes") or []))

        lines.extend(["", "### Good examples (thesis-supporting)", ""])
        good = c.get("good_examples") or []
        if good:
            for ex in good:
                lines.append(_example_block(ex, True))
                lines.append("")
        else:
            lines.append("*None met separation gates for this model.*")
            lines.append("")

        lines.extend(["### Weak examples (honest limits)", ""])
        bad = c.get("bad_examples") or []
        if bad:
            for ex in bad:
                lines.append(_example_block(ex, False))
                lines.append("")
        else:
            lines.append("*No flagged weak examples.*")
            lines.append("")

        lines.extend(["---", ""])

    lines.extend(
        [
            "## Stress gate",
            "",
            "Run `powershell -File scripts/stress_v1.ps1` and `python -m pytest -q -k \"not distilgpt2\"` before trusting this report.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "$env:ANIMA_FORCE_CPU='1'",
            "$env:SKIP_BRAINSCORE='1'",
            "python scripts/run_all_models_benchmark.py",
            "python scripts/generate_benchmark_report.py",
            "```",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
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
            {"model": model_id, "status": "deferred", "reason": reason, "manifest": None, "prompt_examples": []}
        )
    councils: list[dict] = []
    for entry in rollup.get("entries", []):
        manifest = entry.get("manifest")
        if not manifest or not isinstance(manifest, dict):
            continue
        report = score_manifest(manifest, entry.get("prompt_examples") or [])
        out_path = REPORTS / f"council_{entry['model'].replace('/', '_')}.json"
        write_council_json(report, out_path)
        councils.append(report.to_dict())
        print(f"Council {entry['model']}: {report.aggregate_score}/100 passed={report.passed}")

    council_rollup = REPORTS / "council_rollup.json"
    council_rollup.write_text(json.dumps(councils, indent=2), encoding="utf-8")
    md = build_markdown(rollup, councils)
    DOCS.parent.mkdir(parents=True, exist_ok=True)
    DOCS.write_text(md, encoding="utf-8")
    print(f"Wrote {DOCS}")
    try:
        from scripts.generate_benchmark_charts import generate_charts

        chart_paths = generate_charts(rollup, councils, ROOT / "docs" / "images" / "benchmarks")
        for cp in chart_paths:
            print(f"Wrote {cp}")
    except ImportError as exc:
        print(f"Charts skipped (matplotlib?): {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
