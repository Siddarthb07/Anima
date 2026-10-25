#!/usr/bin/env python3
"""Protocol G — free-generation disagreement canary (additive sealed JSON).

This is NOT SelfCheckGPT SOTA. It measures a minimal free-gen instrument:
  for each question, sample N completions; score = mean pairwise disagreement
  (1 - token Jaccard) across samples; AUROC vs HaluEval-style labels when a
  reference incorrect/correct pair exists.

Regime honesty:
  - Claim-encode (Protocol L) teacher-forces answers.
  - Protocol G samples answers; hedging/channels differ by construction.
  - Numbers from this script must be labeled ``protocol_g`` in the paper.

Default CPU canary: distilgpt2, small n. Stage B: instruct models on GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPORTS = Path(__file__).resolve().parent / "reports"


def _auroc(scores: list[float], labels: list[bool]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _disagreement(samples: list[str]) -> float:
    if len(samples) < 2:
        return 0.0
    vals = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            vals.append(1.0 - _jaccard(samples[i], samples[j]))
    return sum(vals) / len(vals)


def _load_halu_questions(n_questions: int, seed: int) -> list[dict[str, Any]]:
    """Return dicts with question + optional right/hall answers for label proxy."""
    from benchmarks.live_guard_eval import load_halueval_pairs

    pairs = load_halueval_pairs(max_pairs=max(n_questions * 2, 40), seed=seed)
    # Collapse to unique questions with a hall label surrogate:
    # we evaluate free-gen risk as ranking questions that have a known hall answer.
    by_q: dict[str, dict[str, Any]] = {}
    for q, a, is_hall in pairs:
        slot = by_q.setdefault(q, {"question": q, "correct": None, "hall": None})
        if is_hall:
            slot["hall"] = a
        else:
            slot["correct"] = a
    rows = [v for v in by_q.values() if v["question"]]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n_questions]


def _generate(model, tokenizer, prompt: str, *, max_new: int, temperature: float, seed: int) -> str:
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5) if do_sample else None,
            top_p=0.9 if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def run_protocol_g(
    *,
    model_name: str = "distilgpt2",
    n_questions: int = 20,
    n_samples: int = 5,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    seed: int = 7,
    device: str = "cpu",
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    questions = _load_halu_questions(n_questions, seed)
    rows: list[dict[str, Any]] = []
    for qi, item in enumerate(questions):
        q = item["question"]
        prompt = f"Answer the question briefly.\nQuestion: {q}\nAnswer:"
        samples = [
            _generate(
                model,
                tokenizer,
                prompt,
                max_new=max_new_tokens,
                temperature=temperature,
                seed=seed + 1000 * qi + s,
            )
            for s in range(n_samples)
        ]
        primary = samples[0]
        # SelfCheck-style score: disagreement of later samples vs primary.
        if len(samples) > 1:
            score = sum(1.0 - _jaccard(primary, s) for s in samples[1:]) / (len(samples) - 1)
        else:
            score = 0.0
        if item.get("hall") and item.get("correct"):
            # Label: primary answer closer to known hall than to correct (proxy).
            label = _jaccard(primary, item["hall"]) >= _jaccard(primary, item["correct"])
            label_def = "primary_closer_to_hall_than_correct_jaccard"
        else:
            label = False
            label_def = "no_pair"
        rows.append(
            {
                "question": q[:240],
                "primary": primary[:200],
                "samples": [s[:200] for s in samples],
                "disagreement": round(score, 4),
                "label": bool(label),
                "label_definition": label_def,
            }
        )

    labels = [r["label"] for r in rows]
    scores = [r["disagreement"] for r in rows]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    auroc = _auroc(scores, labels) if n_pos and n_neg else float("nan")

    return {
        "schema": "anima.protocol_g_free_gen_disagreement",
        "protocol": "G",
        "honesty": {
            "not_selfcheckgpt_sota": True,
            "claim_encode_is_different_regime": True,
            "label_is_sample_hall_proximity_proxy": True,
        },
        "model_name": model_name,
        "n_questions": len(rows),
        "n_samples": n_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": seed,
        "device": device,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auroc_disagreement": None if math.isnan(auroc) else round(float(auroc), 4),
        "mean_disagreement": round(sum(scores) / max(len(scores), 1), 4),
        "rows": rows,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Protocol G free-gen disagreement canary")
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--n-questions", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu")
    p.add_argument("--report-suffix", default="protocol_g")
    args = p.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = run_protocol_g(
        model_name=args.model,
        n_questions=args.n_questions,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        device=args.device,
    )
    slug = args.model.replace("/", "_").replace("-", "_").lower()
    path = REPORTS / f"protocol_g_{slug}_{args.report_suffix}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k != "rows"}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
