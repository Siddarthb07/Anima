"""
Guard-signal ablation study + entropy / fused baselines on claim rows.

Optional semantic-entropy probe: mean SE over free generations (no per-question
hallucination label → reported as summary only, not AUROC).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from benchmarks.guard_metrics import auroc_from_scores, load_fixture, score_abstain_accuracy
from core.guard import ABLATION_SIGNALS, evaluate_guard

REPORTS = Path(__file__).resolve().parent / "reports"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def run_signal_ablations(rows: list[dict[str, Any]], *, label_key: str) -> dict[str, Any]:
    configs: list[tuple[str, tuple[str, ...]]] = [
        ("full", ()),
        ("no_fused", ("fused",)),
        ("no_probe_uncertainty", ("probe_uncertainty",)),
        ("no_hedging", ("hedging",)),
        ("fused_only", ("probe_uncertainty", "hedging")),
        ("probe_only", ("fused", "hedging")),
        ("hedging_only", ("fused", "probe_uncertainty")),
    ]
    out: dict[str, Any] = {"label_key": label_key, "ablations": {}}
    for name, disabled in configs:
        metrics = score_abstain_accuracy(rows, label_key=label_key, disabled_signals=disabled)
        out["ablations"][name] = {"disabled_signals": list(disabled), **metrics}
    return out


def _normalize_answer(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _cluster_entropy(texts: list[str], *, sim_threshold: float = 0.5) -> float:
    norms = [_normalize_answer(t) for t in texts]
    clusters: list[list[str]] = []
    for n in norms:
        placed = False
        for c in clusters:
            if _token_jaccard(n, c[0]) >= sim_threshold:
                c.append(n)
                placed = True
                break
        if not placed:
            clusters.append([n])
    total = max(1, sum(len(c) for c in clusters))
    ent = 0.0
    for c in clusters:
        p = len(c) / total
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    max_ent = math.log(max(2, len(texts)))
    return round(ent / max_ent, 4) if max_ent > 0 else 0.0


def semantic_entropy_for_questions(
    model_name: str,
    questions: list[str],
    *,
    n_samples: int = 5,
    max_new_tokens: int = 24,
    device: str = "cpu",
    temperature: float = 0.8,
) -> list[dict[str, Any]]:
    import torch
    from core.extractor import ActivationExtractor
    from core.prompt_format import format_user_text

    os.environ.setdefault("ANIMA_FORCE_CPU", "1")
    extractor = ActivationExtractor(model_name, device=device)
    results: list[dict[str, Any]] = []
    for qi, question in enumerate(questions):
        prompt = format_user_text(extractor.tokenizer, question, model_name=model_name)
        gens: list[str] = []
        for _ in range(n_samples):
            inputs = extractor.tokenizer(prompt, return_tensors="pt").to(extractor.model.device)
            with torch.no_grad():
                out_ids = extractor.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=extractor.tokenizer.eos_token_id,
                )
            new_tokens = out_ids[0, inputs["input_ids"].shape[1] :]
            gens.append(extractor.tokenizer.decode(new_tokens, skip_special_tokens=True))
        results.append(
            {
                "question": question[:240],
                "semantic_entropy": _cluster_entropy(gens),
                "generations": gens,
            }
        )
        if (qi + 1) % 5 == 0:
            print(f"  semantic-entropy {qi + 1}/{len(questions)}")
    extractor.cleanup()
    return results


def compare_baselines_on_rows(guard_rows: list[dict[str, Any]], *, label_key: str) -> dict[str, Any]:
    labels = [int(bool(r[label_key])) for r in guard_rows]
    composite, fused, entropy = [], [], []
    for r in guard_rows:
        g = evaluate_guard(
            affect=r["affect"],
            uncertainty_signals=r["uncertainty_signals"],
            token_text=r.get("token_text", ""),
        )
        composite.append(g.composite_score)
        fused.append(float(r["uncertainty_signals"].get("fused", 0.5)))
        entropy.append(float(r["uncertainty_signals"].get("entropy", 0.5)))
    return {
        "n": len(guard_rows),
        "auroc_anima_composite": auroc_from_scores(composite, labels),
        "auroc_fused_only": auroc_from_scores(fused, labels),
        "auroc_entropy_only": auroc_from_scores(entropy, labels),
        "abstain_full": score_abstain_accuracy(guard_rows, label_key=label_key),
    }


def _load_or_compute_rows(
    model_name: str,
    *,
    bench: str,
    label_key: str,
    max_pairs: int,
    use_live: bool,
    device: str,
) -> tuple[list[dict[str, Any]], str]:
    slug = model_name.replace("/", "_").replace(".", "_").lower()
    live_path = REPORTS / f"{bench}_live_{slug}.json"
    fixture = FIXTURES / f"{bench}_guard_sample.json"

    if use_live and live_path.is_file():
        rows = json.loads(live_path.read_text(encoding="utf-8")).get("rows") or []
        if rows:
            return rows, "live_cached"

    if use_live:
        from benchmarks.live_guard_eval import (
            evaluate_pairs,
            load_halueval_pairs,
            load_truthfulqa_pairs,
        )

        pairs = (
            load_halueval_pairs(max_pairs=max_pairs)
            if bench == "halueval"
            else load_truthfulqa_pairs(max_pairs=max_pairs)
        )
        rows = evaluate_pairs(model_name, pairs, label_key=label_key, device=device)
        REPORTS.mkdir(parents=True, exist_ok=True)
        live_path.write_text(
            json.dumps({"rows": rows, "metrics": score_abstain_accuracy(rows, label_key=label_key)}, indent=2),
            encoding="utf-8",
        )
        return rows, "live_fresh"

    return load_fixture(fixture), "fixture"


def run_ablation_suite(
    *,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    use_live_rows: bool = True,
    max_pairs: int = 40,
    run_semantic: bool = True,
    semantic_questions: int = 15,
    device: str = "cpu",
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "ok",
        "model": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "available_ablation_signals": list(ABLATION_SIGNALS),
        "halueval": {},
        "truthfulqa": {},
        "baselines": {},
    }
    slug = model_name.replace("/", "_").replace(".", "_").lower()

    for bench, label_key in (
        ("halueval", "is_hallucination"),
        ("truthfulqa", "should_abstain"),
    ):
        print(f"[ablation] {bench} rows…")
        rows, source = _load_or_compute_rows(
            model_name,
            bench=bench,
            label_key=label_key,
            max_pairs=max_pairs,
            use_live=use_live_rows,
            device=device,
        )
        ab = run_signal_ablations(rows, label_key=label_key)
        ab["source"] = source
        ab["n_rows"] = len(rows)
        report[bench] = ab
        if bench == "halueval":
            report["baselines"] = compare_baselines_on_rows(rows, label_key=label_key)
            report["baselines"]["source"] = source

    if run_semantic:
        from benchmarks.live_guard_eval import load_halueval_pairs

        pairs = load_halueval_pairs(max_pairs=max(semantic_questions * 2, 20))
        uniq: list[str] = []
        seen: set[str] = set()
        for q, _a, _lab in pairs:
            if q not in seen:
                seen.add(q)
                uniq.append(q)
            if len(uniq) >= semantic_questions:
                break
        print(f"[ablation] semantic entropy on {len(uniq)} questions")
        se_rows = semantic_entropy_for_questions(model_name, uniq, n_samples=5, device=device)
        mean_se = sum(r["semantic_entropy"] for r in se_rows) / max(1, len(se_rows))
        report["semantic_entropy_summary"] = {
            "n_questions": len(se_rows),
            "mean_semantic_entropy": round(mean_se, 4),
            "note": (
                "Mean SE over free generations; no per-question hallucination label, "
                "so AUROC is not claimed. Claim-level AUROCs are under baselines/ablations."
            ),
        }
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / f"semantic_entropy_{slug}.json").write_text(
            json.dumps({"rows": se_rows, "mean": mean_se}, indent=2), encoding="utf-8"
        )

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"guard_ablation_{slug}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return report


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Guard ablations + semantic-entropy baseline")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max-pairs", type=int, default=40)
    p.add_argument("--no-live", action="store_true")
    p.add_argument("--no-semantic", action="store_true")
    p.add_argument("--semantic-questions", type=int, default=15)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    run_ablation_suite(
        model_name=args.model,
        use_live_rows=not args.no_live,
        max_pairs=args.max_pairs,
        run_semantic=not args.no_semantic,
        semantic_questions=args.semantic_questions,
        device=args.device,
    )


if __name__ == "__main__":
    main()
