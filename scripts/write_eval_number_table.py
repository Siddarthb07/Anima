"""Write a legible markdown number table from locked + L′ artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPORTS = Path(__file__).resolve().parent.parent / "benchmarks" / "reports"
NOTES = Path(__file__).resolve().parent.parent / "docs" / "notes"


def _j(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def build_table(reports: Path = REPORTS) -> str:
    models = [
        ("qwen_qwen2_5-0_5b-instruct", "Qwen2.5-0.5B-Instruct"),
        ("tinyllama_tinyllama-1_1b-chat-v1_0", "TinyLlama-1.1B-Chat"),
    ]
    lines = [
        "# Anima locked eval — complete number table",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "Honesty: claim-encoding on public HaluEval/TruthfulQA rows; not free-gen+judge; "
        "not hallucination SOTA; misleading-green is simulated badge replay (not a user study). "
        "Baseline Protocol L is the collapse finding. Protocol L′ is an explicit LOQO ridge "
        "adaptation (production `guard_config.yaml` unchanged; no silent label flips).",
        "",
        "## A. Protocol F (fixture smoke)",
        "",
        "| Suite | n | AUROC composite | Note |",
        "|-------|---:|----------------:|------|",
        "| HaluEval fixture | 52 | 1.0000 | CI smoke only — not citeable as detection |",
        "| TruthfulQA fixture | 52 | 1.0000 | CI smoke only — not citeable as detection |",
        "",
        "## B. Protocol L — baseline live (default composite)",
        "",
        "| Model | Suite | n | AUROC | 95% CI | Never-abstain | Polarity | Δ F−L |",
        "|-------|-------|---:|------:|--------|---------------|----------|------:|",
    ]

    for slug, name in models:
        live = _j(reports / f"live_guard_{slug}.json") or {}
        pol = _j(reports / f"polarity_diagnostic_{slug}.json") or {}
        for suite in ("halueval", "truthfulqa"):
            b = (live.get("benchmarks") or {}).get(suite) or {}
            # fallback metrics file
            if not b:
                rowdoc = _j(reports / f"{suite}_live_{slug}.json") or {}
                b = rowdoc.get("metrics") or {}
            ci = b.get("auroc_ci95") or {}
            ci_s = (
                f"[{_fmt(ci.get('lo'), 3)}, {_fmt(ci.get('hi'), 3)}]"
                if ci.get("lo") is not None
                else "—"
            )
            auroc = b.get("auroc_composite")
            gap = None if auroc is None else round(1.0 - float(auroc), 4)
            verdict = ((pol.get("suites") or {}).get(suite) or {}).get("verdict", "—")
            lines.append(
                f"| {name} | {suite} | {_fmt(b.get('n_samples'), 0)} | {_fmt(auroc)} | {ci_s} | "
                f"{_fmt(b.get('never_abstains'))} | {verdict} | {_fmt(gap)} |"
            )

    lines.extend(
        [
            "",
            "## C. Protocol R / R′ (threshold retune on locked rows)",
            "",
            "| Model | Suite | R best abstain acc | R′ test abstain acc |",
            "|-------|-------|-------------------:|--------------------:|",
        ]
    )
    for slug, name in models:
        r = _j(reports / f"protocol_r_{slug}.json") or {}
        rp_file = _j(reports / f"protocol_r_prime_{slug}.json") or {}
        for suite in ("halueval", "truthfulqa"):
            rb = (r.get("suites") or {}).get(suite) or {}
            # R and R′ often nest inside protocol_r_*.json; prime file is fallback
            r_block = rb.get("protocol_r") or rb
            rp_block = rb.get("protocol_r_prime") or (
                (rp_file.get("suites") or {}).get(suite) or {}
            )
            r_acc = r_block.get("best_acc") or r_block.get("best_abstain_acc")
            rp_acc = rp_block.get("test_acc") or rp_block.get("test_abstain_acc")
            lines.append(f"| {name} | {suite} | {_fmt(r_acc)} | {_fmt(rp_acc)} |")

    lines.extend(
        [
            "",
            "## D. Matched semantic entropy (same Protocol L questions)",
            "",
            "| Model | Suite | SE AUROC | Claim-disagreement | Anima composite | Probe-only |",
            "|-------|-------|---------:|-------------------:|----------------:|-----------:|",
        ]
    )
    for slug, name in models:
        se = _j(reports / f"semantic_entropy_matched_{slug}.json") or {}
        for suite in ("halueval", "truthfulqa"):
            b = (se.get("suites") or {}).get(suite) or {}
            base = b.get("baselines") or {}
            lines.append(
                f"| {name} | {suite} | "
                f"{_fmt((base.get('semantic_entropy') or {}).get('auroc'))} | "
                f"{_fmt((base.get('claim_disagreement') or {}).get('auroc'))} | "
                f"{_fmt((base.get('anima_composite') or {}).get('auroc'))} | "
                f"{_fmt((base.get('probe_only') or {}).get('auroc'))} |"
            )

    lines.extend(
        [
            "",
            "## E. Protocol L′ — LOQO ridge adaptation (unlock operating point)",
            "",
            "| Model | Suite | LOQO AUROC | 95% CI | Same-scorer F′ | Δ F′−L′ | MGR | Abstain | Polarity |",
            "|-------|-------|-----------:|--------|---------------:|--------:|----:|--------:|----------|",
        ]
    )
    for slug, name in models:
        lp = _j(reports / f"protocol_l_prime_{slug}.json") or {}
        for suite in ("halueval", "truthfulqa"):
            b = (lp.get("suites") or {}).get(suite) or {}
            if b.get("status") != "ok":
                lines.append(f"| {name} | {suite} | — | — | — | — | — | — | missing |")
                continue
            ci = b.get("auroc_ci95") or {}
            ci_s = f"[{_fmt(ci.get('lo'), 3)}, {_fmt(ci.get('hi'), 3)}]"
            mgr = (b.get("misleading_green") or {}).get("misleading_green_rate")
            lines.append(
                f"| {name} | {suite} | {_fmt(b.get('auroc_loqo'))} | {ci_s} | "
                f"{_fmt(b.get('protocol_f_same_scorer_auroc'))} | "
                f"{_fmt(b.get('transfer_gap_same_scorer'))} | {_fmt(mgr)} | "
                f"{b.get('n_abstain')}/{b.get('n')} | {b.get('polarity_verdict')} |"
            )

    lines.extend(
        [
            "",
            "## F. Trust Integrity Gate + GreenShieldBench",
            "",
            "| Model | OP | claim_allowed | GSB passed | Failed gates (short) |",
            "|-------|----|---------------|------------|----------------------|",
        ]
    )
    for slug, name in models:
        for op, tig_name, gsb_name in (
            ("baseline L", f"trust_integrity_{slug}.json", f"greenshield_bench_{slug}.json"),
            (
                "adapted L′",
                f"trust_integrity_adapted_{slug}.json",
                f"greenshield_bench_adapted_{slug}.json",
            ),
        ):
            tig = _j(reports / tig_name) or {}
            gsb = _j(reports / gsb_name) or {}
            fails = tig.get("failed_gates") or gsb.get("failed_checks") or []
            short = ", ".join(fails[:4]) + ("…" if len(fails) > 4 else "")
            lines.append(
                f"| {name} | {op} | {_fmt(tig.get('trust_ui_claim_allowed'))} | "
                f"{_fmt(gsb.get('passed'))} | {short or '(none)'} |"
            )

    lines.extend(
        [
            "",
            "## G. What is / is not claimed",
            "",
            "| Claim | Status |",
            "|-------|--------|",
            "| Fixture AUROC 1.00 proves hallucination detection | **False** — smoke only |",
            "| Baseline live default composite is production-ready | **False** — locked FAIL |",
            "| L′ LOQO ridge unlocks trust-UI under named adaptation | See table F adapted rows |",
            "| Brain / TRIBEv2 voxel alignment | **Not claimed** |",
            "| User-study proof of misleading badges | **Not claimed** — simulated MGR only |",
            "| Hallucination SOTA | **Not claimed** |",
            "",
            "---",
            "",
            "*Not a paper draft. Numbers must match JSON under `benchmarks/reports/`.*",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    p.add_argument("--out", type=Path, default=NOTES / "ANIMA_NUMBER_TABLE.md")
    args = p.parse_args(argv)
    text = build_table(args.reports_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    (args.reports_dir / "ANIMA_NUMBER_TABLE.md").write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.reports_dir / 'ANIMA_NUMBER_TABLE.md'}")


if __name__ == "__main__":
    main()
