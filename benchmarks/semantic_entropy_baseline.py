"""
A2 — Matched semantic-entropy baseline on the same Protocol L claim rows.

For each locked live row (question, answer, label):
  - free-generate n_samples answers from the question (Kuhn-style)
  - semantic_entropy = normalized cluster entropy of generations
  - claim_disagreement = 1 − match_frac vs the given answer (claim-specific)

Reports AUROC (+ bootstrap CI) for SE and disagreement vs composite / probe-only
on the identical row set (IEEE: same n, same labels, same seed meta).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from benchmarks.guard_metrics import auroc_from_scores, bootstrap_auroc_ci
from benchmarks.semantic_entropy import claim_disagreement, cluster_entropy
from core.guard import evaluate_guard

REPORTS = Path(__file__).resolve().parent / "reports"


def _label_key_for_suite(suite: str) -> str:
    return "is_hallucination" if suite == "halueval" else "should_abstain"


def _probe_scores(rows: list[dict[str, Any]]) -> list[float]:
    return [float((r.get("affect") or {}).get("uncertainty", 0.5)) for r in rows]


def _composite_scores(rows: list[dict[str, Any]]) -> list[float]:
    out = []
    for r in rows:
        g = evaluate_guard(
            affect=r["affect"],
            uncertainty_signals=r["uncertainty_signals"],
            token_text=r.get("token_text", ""),
        )
        out.append(g.composite_score)
    return out


def generate_for_questions(
    model_name: str,
    questions: list[str],
    *,
    n_samples: int = 5,
    max_new_tokens: int = 24,
    device: str = "cpu",
    temperature: float = 0.8,
) -> dict[str, list[str]]:
    """Cache free-gens keyed by exact question string."""
    import torch
    from core.extractor import ActivationExtractor
    from core.prompt_format import format_user_text

    os.environ.setdefault("ANIMA_FORCE_CPU", "1")
    print(f"[SE] loading model {model_name} on {device}…", flush=True)
    extractor = ActivationExtractor(model_name, device=device)
    print("[SE] model ready", flush=True)
    cache: dict[str, list[str]] = {}
    uniq = list(dict.fromkeys(questions))
    for qi, question in enumerate(uniq):
        prompt = format_user_text(extractor.tokenizer, question, model_name=model_name)
        gens: list[str] = []
        for _ in range(n_samples):
            inputs = extractor.tokenizer(prompt, return_tensors="pt").to(extractor.model.device)
            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=extractor.tokenizer.eos_token_id,
                )
                # Hard cap wall-clock per sample so CPU runs cannot hang forever.
                try:
                    out_ids = extractor.model.generate(**inputs, **gen_kwargs, max_time=30.0)
                except TypeError:
                    out_ids = extractor.model.generate(**inputs, **gen_kwargs)
            new_tokens = out_ids[0, inputs["input_ids"].shape[1] :]
            gens.append(extractor.tokenizer.decode(new_tokens, skip_special_tokens=True))
        cache[question] = gens
        print(f"  SE gens {qi + 1}/{len(uniq)}", flush=True)
    extractor.cleanup()
    return cache


def score_suite_from_rows(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    gen_cache: dict[str, list[str]],
    B: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    labels = [int(bool(r[label_key])) for r in rows]
    se_scores: list[float] = []
    disagree: list[float] = []
    detail: list[dict[str, Any]] = []
    for r in rows:
        q = str(r.get("question") or "")
        a = str(r.get("answer") or "")
        gens = gen_cache.get(q) or []
        se = cluster_entropy(gens) if gens else 0.5
        disc = claim_disagreement(gens, a) if gens else 0.5
        se_scores.append(se)
        disagree.append(disc)
        detail.append(
            {
                "question": q[:240],
                "answer": a[:240],
                label_key: bool(r[label_key]),
                "semantic_entropy": se,
                "claim_disagreement": disc,
                "n_generations": len(gens),
            }
        )
    composite = _composite_scores(rows)
    probe = _probe_scores(rows)

    def pack(name: str, scores: list[float]) -> dict[str, Any]:
        return {
            "auroc": auroc_from_scores(scores, labels),
            "auroc_ci95": bootstrap_auroc_ci(scores, labels, B=B, seed=seed),
        }

    return {
        "n": len(rows),
        "label_key": label_key,
        "baselines": {
            "anima_composite": pack("composite", composite),
            "probe_only": pack("probe", probe),
            "semantic_entropy": pack("se", se_scores),
            "claim_disagreement": pack("disagree", disagree),
        },
        "ieee_note": (
            "claim_disagreement is the primary matched SE-family score for paired "
            "correct/hall claims; raw SE alone is question-constant within pairs."
        ),
        "rows": detail,
    }


def run_matched(
    *,
    model_name: str,
    slug: Optional[str] = None,
    reports_dir: Path = REPORTS,
    n_samples: int = 5,
    device: str = "cpu",
    B: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    slug = slug or model_name.replace("/", "_").replace(".", "_").lower()
    report: dict[str, Any] = {
        "status": "ok",
        "protocol": "L_matched_semantic_entropy",
        "model": model_name,
        "model_slug": slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_samples_per_question": n_samples,
        "seed": seed,
        "B": B,
        "suites": {},
    }
    # Collect questions across suites for one generation pass
    suite_rows: dict[str, list[dict[str, Any]]] = {}
    all_questions: list[str] = []
    for suite in ("halueval", "truthfulqa"):
        path = reports_dir / f"{suite}_live_{slug}.json"
        if not path.is_file():
            report["suites"][suite] = {"status": "missing", "path": str(path)}
            continue
        rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
        suite_rows[suite] = rows
        all_questions.extend(str(r.get("question") or "") for r in rows)

    if not suite_rows:
        report["status"] = "error"
        report["error"] = "no live row files found"
        return report

    print(
        f"[SE] generating for {len(set(all_questions))} unique questions on {model_name}",
        flush=True,
    )
    cache = generate_for_questions(
        model_name, all_questions, n_samples=n_samples, device=device
    )

    for suite, rows in suite_rows.items():
        label_key = _label_key_for_suite(suite)
        block = score_suite_from_rows(
            rows, label_key=label_key, gen_cache=cache, B=B, seed=seed
        )
        block["status"] = "ok"
        report["suites"][suite] = block
        b = block["baselines"]
        print(
            f"  {suite}: SE={b['semantic_entropy']['auroc']} "
            f"disagree={b['claim_disagreement']['auroc']} "
            f"composite={b['anima_composite']['auroc']} "
            f"probe={b['probe_only']['auroc']}",
            flush=True,
        )
    return report


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Matched semantic-entropy baseline (Protocol L rows)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--slug", default=None)
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    args = p.parse_args(argv)
    report = run_matched(
        model_name=args.model,
        slug=args.slug,
        reports_dir=args.reports_dir,
        n_samples=args.n_samples,
        device=args.device,
        B=args.B,
        seed=args.seed,
    )
    slug = report.get("model_slug") or "unknown"
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    out = args.reports_dir / f"semantic_entropy_matched_{slug}.json"
    # drop bulky generations from summary write — detail rows already omit gens
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
