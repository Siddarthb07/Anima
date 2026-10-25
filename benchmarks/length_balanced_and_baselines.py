#!/usr/bin/env python3
"""Length-balanced + multi-baseline table from locked rows (additive; no overwrite).

Designs:
A) Length-bin balanced: within each answer-length quartile, downsample to equal
   pos/neg, then pool (breaks some pairs; controls length mass).
B) Cross-question greedy length match: match each positive to nearest-length
   unused negative (unpaired control).
C) Baseline AUROCs: composite, probe, fused, inv logit-gap, answer length,
   residualized composite, random (seeded).
D) Include DistilGPT2 locked rows as third architecture canary if present.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from benchmarks.pairing_bootstrap_audit import auroc, clustered_bootstrap, score_of

REPORTS = Path(__file__).resolve().parent / "reports"

MODELS = [
    ("qwen_qwen2_5-0_5b-instruct", "Qwen2.5-0.5B"),
    ("tinyllama_tinyllama-1_1b-chat-v1_0", "TinyLlama-1.1B"),
    ("distilgpt2", "DistilGPT2"),
]


def inv_gap(row: dict[str, Any]) -> float:
    return 1.0 - float(row["uncertainty_signals"]["logit_gap"])


def residualized_composite(rows: list[dict[str, Any]]) -> list[float]:
    lens = [float(len(str(r.get("answer") or ""))) for r in rows]
    comps = [score_of(r, "composite") for r in rows]
    n = len(rows)
    mx = sum(lens) / n
    my = sum(comps) / n
    varx = sum((x - mx) ** 2 for x in lens) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(lens, comps)) / n
    slope = cov / varx if varx > 0 else 0.0
    return [y - (my + slope * (x - mx)) for x, y in zip(lens, comps)]


def length_bin_balanced(rows: list[dict[str, Any]], label_key: str, *, seed: int = 42) -> list[dict[str, Any]]:
    lens = [len(str(r.get("answer") or "")) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: lens[i])
    n = len(rows)
    kept: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for t in range(4):
        idx = order[t * n // 4 : (t + 1) * n // 4]
        pos = [i for i in idx if rows[i][label_key]]
        neg = [i for i in idx if not rows[i][label_key]]
        m = min(len(pos), len(neg))
        if m == 0:
            continue
        rng.shuffle(pos)
        rng.shuffle(neg)
        for i in pos[:m] + neg[:m]:
            kept.append(rows[i])
    return kept


def greedy_length_match(rows: list[dict[str, Any]], label_key: str) -> list[dict[str, Any]]:
    pos = [r for r in rows if r[label_key]]
    neg = [r for r in rows if not r[label_key]]
    neg_used = set()
    kept: list[dict[str, Any]] = []
    for p in sorted(pos, key=lambda r: len(str(r.get("answer") or ""))):
        lp = len(str(p.get("answer") or ""))
        best_j = None
        best_d = 10**9
        for j, n in enumerate(neg):
            if j in neg_used:
                continue
            d = abs(len(str(n.get("answer") or "")) - lp)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is None:
            break
        neg_used.add(best_j)
        kept.append(p)
        kept.append(neg[best_j])
    return kept


def pack_scores(rows: list[dict[str, Any]], label_key: str) -> dict[str, Any]:
    labs = [int(bool(r[label_key])) for r in rows]
    resid = residualized_composite(rows)
    rng = random.Random(0)
    rnd = [rng.random() for _ in rows]
    out = {
        "n": len(rows),
        "n_pos": sum(labs),
        "n_neg": len(labs) - sum(labs),
        "baselines": {
            "composite": round(auroc([score_of(r, "composite") for r in rows], labs), 4),
            "probe": round(auroc([score_of(r, "probe") for r in rows], labs), 4),
            "fused": round(auroc([score_of(r, "fused") for r in rows], labs), 4),
            "inv_logit_gap": round(auroc([inv_gap(r) for r in rows], labs), 4),
            "answer_length": round(auroc([float(len(str(r.get("answer") or ""))) for r in rows], labs), 4),
            "composite_residualized_on_length": round(auroc(resid, labs), 4),
            "random": round(auroc(rnd, labs), 4),
        },
    }
    if len(set(labs)) == 2 and len(rows) >= 20:
        out["cluster_bootstrap_composite"] = clustered_bootstrap(rows, kind="composite", label_key=label_key)
    return out


def load_rows(suite: str, slug: str) -> list[dict[str, Any]] | None:
    p = REPORTS / f"{suite}_live_{slug}.json"
    if not p.is_file():
        return None
    return list(json.loads(p.read_text(encoding="utf-8"))["rows"])


def main() -> None:
    report: dict[str, Any] = {
        "schema": "anima.length_balanced_and_baselines",
        "status": "ok",
        "honesty": {"locked_n80_untouched": True, "additive": True},
        "models": {},
    }
    for slug, short in MODELS:
        block: dict[str, Any] = {"short": short, "suites": {}}
        for suite, lk in (("halueval", "is_hallucination"), ("truthfulqa", "should_abstain")):
            rows = load_rows(suite, slug)
            if not rows:
                continue
            full = pack_scores(rows, lk)
            binbal = length_bin_balanced(rows, lk)
            greedy = greedy_length_match(rows, lk)
            block["suites"][suite] = {
                "full_locked": full,
                "length_bin_balanced": pack_scores(binbal, lk) if len(binbal) >= 20 else {"n": len(binbal), "status": "too_small"},
                "greedy_cross_question_length_match": pack_scores(greedy, lk) if len(greedy) >= 20 else {"n": len(greedy), "status": "too_small"},
            }
            print(
                short,
                suite,
                "full",
                full["baselines"],
                "binbal",
                block["suites"][suite]["length_bin_balanced"],
            )
        if block["suites"]:
            report["models"][slug] = block

    out = REPORTS / "length_balanced_and_baselines.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
