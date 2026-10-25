#!/usr/bin/env python3
"""Additive analyses toward 9/10 readiness (does NOT overwrite locked n=80 baselines).

1) Risk-coverage curves from locked live rows (composite / probe / fused / -length).
2) Length-matched HaluEval: keep only question-pairs whose |len(hall)-len(good)| <= tol,
   or subsample pairs in matched length bins.
3) Report AUROC + clustered bootstrap on the matched subset.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.pairing_bootstrap_audit import auroc, clustered_bootstrap, score_of

REPORTS = Path(__file__).resolve().parent / "reports"
SLUGS = (
    "qwen_qwen2_5-0_5b-instruct",
    "tinyllama_tinyllama-1_1b-chat-v1_0",
)


def risk_coverage(scores: list[float], labels: list[int], *, higher_is_risk: bool = True) -> list[dict[str, float]]:
    """Selective prediction: abstain on highest-risk fraction; report coverage vs error on answered."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i], reverse=higher_is_risk)
    out = []
    for k in range(0, n + 1, max(1, n // 20)):
        # abstain top-k riskiest; answer the rest
        answered = [i for i in range(n) if i not in set(order[:k])]
        cov = len(answered) / n
        if not answered:
            out.append({"abstain_frac": k / n, "coverage": 0.0, "error_on_answered": None})
            continue
        # error = predicting "safe/correct" when label positive (hall / should_abstain)
        # Under claim-encode we don't have a separate answer correctness model; use label as risk.
        # Risk-coverage for ranking: among answered (low score), fraction that are positive labels.
        err = sum(labels[i] for i in answered) / len(answered)
        out.append(
            {
                "abstain_frac": round(k / n, 4),
                "coverage": round(cov, 4),
                "positive_rate_on_answered": round(err, 4),
            }
        )
    return out


def length_matched_halueval(rows: list[dict[str, Any]], *, max_len_ratio: float = 1.5) -> list[dict[str, Any]]:
    """Keep pairs where longer answer / shorter answer <= max_len_ratio."""
    byq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        byq[str(r["question"])].append(r)
    kept: list[dict[str, Any]] = []
    for rs in byq.values():
        if len(rs) != 2:
            continue
        a, b = rs
        if bool(a["is_hallucination"]) == bool(b["is_hallucination"]):
            continue
        la, lb = len(str(a.get("answer") or "")), len(str(b.get("answer") or ""))
        hi, lo = max(la, lb), max(1, min(la, lb))
        if hi / lo <= max_len_ratio:
            kept.extend(rs)
    return kept


def main() -> None:
    report: dict[str, Any] = {
        "schema": "anima.risk_coverage_and_length_match",
        "status": "ok",
        "honesty": {
            "baseline_n80_untouched": True,
            "additive_only": True,
        },
        "models": {},
    }
    for slug in SLUGS:
        model_block: dict[str, Any] = {"model_slug": slug, "suites": {}}
        for suite, lk in (("halueval", "is_hallucination"), ("truthfulqa", "should_abstain")):
            rows = json.loads((REPORTS / f"{suite}_live_{slug}.json").read_text(encoding="utf-8"))["rows"]
            labs = [int(bool(r[lk])) for r in rows]
            block: dict[str, Any] = {
                "n": len(rows),
                "risk_coverage": {
                    "composite": risk_coverage([score_of(r, "composite") for r in rows], labs),
                    "probe": risk_coverage([score_of(r, "probe") for r in rows], labs),
                    "fused": risk_coverage([score_of(r, "fused") for r in rows], labs),
                    "answer_length": risk_coverage(
                        [float(len(str(r.get("answer") or ""))) for r in rows], labs
                    ),
                    "minus_answer_length": risk_coverage(
                        [-float(len(str(r.get("answer") or ""))) for r in rows], labs
                    ),
                },
            }
            if suite == "halueval":
                matched = length_matched_halueval(rows, max_len_ratio=1.5)
                matched_strict = length_matched_halueval(rows, max_len_ratio=1.25)
                for name, subset in (("ratio_1_5", matched), ("ratio_1_25", matched_strict)):
                    if len(subset) < 20:
                        block[f"length_matched_{name}"] = {"n": len(subset), "status": "too_small"}
                        continue
                    mlabs = [int(bool(r[lk])) for r in subset]
                    block[f"length_matched_{name}"] = {
                        "n": len(subset),
                        "n_questions": len({r["question"] for r in subset}),
                        "auroc_composite": round(auroc([score_of(r, "composite") for r in subset], mlabs), 4),
                        "auroc_probe": round(auroc([score_of(r, "probe") for r in subset], mlabs), 4),
                        "auroc_fused": round(auroc([score_of(r, "fused") for r in subset], mlabs), 4),
                        "auroc_length": round(
                            auroc([float(len(str(r.get("answer") or ""))) for r in subset], mlabs), 4
                        ),
                        "cluster_bootstrap_composite": clustered_bootstrap(
                            subset, kind="composite", label_key=lk
                        ),
                    }
            model_block["suites"][suite] = block
            print(slug, suite, "n", len(rows))
            if suite == "halueval":
                for name in ("length_matched_ratio_1_5", "length_matched_ratio_1_25"):
                    print(" ", name, block.get(name))
        report["models"][slug] = model_block

    out = REPORTS / "risk_coverage_and_length_match.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
