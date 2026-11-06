#!/usr/bin/env python3
"""Groq-backed free-gen disagreement canary (Wave-3 GPU bypass).

Uses Groq OpenAI-compatible Chat Completions when GROQ_API_KEY is set.
Regime: Protocol G / free-gen — NOT claim-encode; NOT a SelfCheckGPT SOTA claim.

  export GROQ_API_KEY=...
  python -m benchmarks.groq_free_gen --n-questions 20 --n-samples 4

Default model: llama-3.1-8b-instant (fast). shipping_claim=false.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from benchmarks.free_gen_protocol import _auroc, _disagreement, _load_halu_questions

REPORTS = Path(__file__).resolve().parent / "reports"
PAPER_ART = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact\prereg")


def _groq_client():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "GROQ_API_KEY not set. Paste a Groq key then re-run — this bypasses "
            "local A10G rental for free-gen SE/SelfCheck-style sampling."
        )
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("pip install openai  # Groq uses OpenAI-compatible client") from e
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")


def sample_answers(
    client: Any,
    question: str,
    *,
    model: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
) -> list[str]:
    out: list[str] = []
    for _ in range(n_samples):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Answer the question in one short sentence.\n\n"
                        f"Question: {question}"
                    ),
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        out.append(text)
        time.sleep(0.05)
    return out


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="llama-3.1-8b-instant")
    p.add_argument("--n-questions", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=REPORTS / "groq_free_gen_canary.json")
    p.add_argument("--also-paper", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="No API calls; write stub receipt")
    args = p.parse_args(argv)

    if args.dry_run:
        payload = {
            "schema": "anima.groq_free_gen.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "awaiting_GROQ_API_KEY",
            "shipping_claim": False,
            "model": args.model,
            "notes": "Set GROQ_API_KEY and re-run without --dry-run for Wave-3 free-gen.",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote dry-run {args.out}")
        return

    client = _groq_client()
    questions = _load_halu_questions(args.n_questions, args.seed)
    rows = []
    labels = []
    scores = []
    for qrow in questions:
        q = qrow["question"]
        samples = sample_answers(
            client,
            q,
            model=args.model,
            n_samples=args.n_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        score = _disagreement(samples)
        # Label proxy: question has a known hall answer in HaluEval collapse
        is_hall = bool(qrow.get("hall"))
        rows.append(
            {
                "question": q,
                "samples": samples,
                "disagreement": score,
                "label_proxy_has_hall": is_hall,
            }
        )
        scores.append(score)
        labels.append(is_hall)

    payload = {
        "schema": "anima.groq_free_gen.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shipping_claim": False,
        "regime": "protocol_g_free_gen_groq",
        "model": args.model,
        "n_questions": len(rows),
        "n_samples": args.n_samples,
        "auroc_disagreement_vs_has_hall": _auroc(scores, labels),
        "rows": rows,
        "notes": "Not SelfCheckGPT SOTA; sample-disagreement canary via Groq.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out} auroc={payload['auroc_disagreement_vs_has_hall']}")
    if args.also_paper:
        PAPER_ART.mkdir(parents=True, exist_ok=True)
        (PAPER_ART / args.out.name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
