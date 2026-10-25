#!/usr/bin/env python3
"""Pairing-artifact + question-clustered bootstrap audit (additive; no baseline overwrite)."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

REPORTS = Path(__file__).resolve().parent / "reports"
SLUGS = (
    "qwen_qwen2_5-0_5b-instruct",
    "tinyllama_tinyllama-1_1b-chat-v1_0",
)


def auroc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def score_of(row: dict[str, Any], kind: str) -> float:
    if kind == "composite":
        if "composite_score" in row and row["composite_score"] is not None:
            return float(row["composite_score"])
        # fallback: 0.55 fused + 0.45 probe
        fused = float(row["uncertainty_signals"]["fused"])
        probe = float(row["affect"]["uncertainty"])
        return 0.55 * fused + 0.45 * probe
    if kind == "fused":
        return float(row["uncertainty_signals"]["fused"])
    if kind == "probe":
        return float(row["affect"]["uncertainty"])
    raise KeyError(kind)


def clustered_bootstrap(
    rows: list[dict[str, Any]],
    *,
    kind: str,
    label_key: str,
    B: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r["question"])].append(r)
    keys = list(groups.keys())
    scores = [score_of(r, kind) for r in rows]
    labels = [int(bool(r[label_key])) for r in rows]
    point = auroc(scores, labels)
    rng = random.Random(seed)
    vals: list[float] = []
    for _ in range(B):
        samp = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        srows: list[dict[str, Any]] = []
        for k in samp:
            srows.extend(groups[k])
        sc = [score_of(r, kind) for r in srows]
        lab = [int(bool(r[label_key])) for r in srows]
        vals.append(0.5 if len(set(lab)) < 2 else auroc(sc, lab))
    vals.sort()
    return {
        "point": round(point, 4),
        "lo": round(vals[int(0.025 * B)], 4),
        "hi": round(vals[min(B - 1, int(0.975 * B))], 4),
        "B": B,
        "seed": seed,
        "method": "question_cluster_bootstrap",
        "n_questions": len(keys),
        "n_rows": len(rows),
    }


def suite_audit(suite: str, slug: str) -> dict[str, Any]:
    label_key = "is_hallucination" if suite == "halueval" else "should_abstain"
    path = REPORTS / f"{suite}_live_{slug}.json"
    rows = list(json.loads(path.read_text(encoding="utf-8"))["rows"])
    byq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        byq[str(r["question"])].append(r)
    size_hist: dict[int, int] = {}
    for rs in byq.values():
        size_hist[len(rs)] = size_hist.get(len(rs), 0) + 1

    pos = [r for r in rows if r[label_key]]
    neg = [r for r in rows if not r[label_key]]

    def mean_len(rs: list[dict[str, Any]], field: str) -> float:
        vals = [len(str(r.get(field) or "")) for r in rs]
        return round(sum(vals) / max(1, len(vals)), 2)

    ans_len = [len(str(r.get("answer") or "")) for r in rows]
    labs = [int(bool(r[label_key])) for r in rows]
    len_auroc = round(auroc(ans_len, labs), 4)

    deltas: list[float] = []
    len_deltas: list[float] = []
    for rs in byq.values():
        if len(rs) != 2:
            continue
        a, b = rs
        if bool(a[label_key]) == bool(b[label_key]):
            continue
        hall = a if a[label_key] else b
        good = b if a[label_key] else a
        deltas.append(score_of(hall, "composite") - score_of(good, "composite"))
        len_deltas.append(len(str(hall.get("answer") or "")) - len(str(good.get("answer") or "")))

    return {
        "label_key": label_key,
        "n_rows": len(rows),
        "n_questions": len(byq),
        "question_group_size_hist": {str(k): v for k, v in sorted(size_hist.items())},
        "answer_length": {
            "pos_mean": mean_len(pos, "answer"),
            "neg_mean": mean_len(neg, "answer"),
            "pos_token_text_mean": mean_len(pos, "token_text"),
            "neg_token_text_mean": mean_len(neg, "token_text"),
            "auroc_label_from_answer_length_alone": len_auroc,
        },
        "within_pair": {
            "n_pairs": len(deltas),
            "composite_delta_hall_minus_correct_mean": round(sum(deltas) / len(deltas), 4) if deltas else None,
            "frac_hall_higher_composite": round(sum(1 for d in deltas if d > 0) / len(deltas), 4) if deltas else None,
            "frac_hall_lower_composite": round(sum(1 for d in deltas if d < 0) / len(deltas), 4) if deltas else None,
            "answer_len_delta_hall_minus_correct_mean": round(sum(len_deltas) / len(len_deltas), 2) if len_deltas else None,
        },
        "question_cluster_bootstrap": {
            "composite": clustered_bootstrap(rows, kind="composite", label_key=label_key),
            "probe": clustered_bootstrap(rows, kind="probe", label_key=label_key),
            "fused": clustered_bootstrap(rows, kind="fused", label_key=label_key),
        },
        "honesty": {
            "hedging_null_by_construction_under_claim_encode": True,
            "claim_encode_is_not_free_generation": True,
        },
    }


def main() -> None:
    out: dict[str, Any] = {
        "schema": "anima.pairing_and_paired_bootstrap_audit",
        "status": "ok",
        "models": {},
    }
    for slug in SLUGS:
        suites = {}
        for suite in ("halueval", "truthfulqa"):
            suites[suite] = suite_audit(suite, slug)
            c = suites[suite]["question_cluster_bootstrap"]["composite"]
            w = suites[suite]["within_pair"]
            a = suites[suite]["answer_length"]
            print(
                f"{slug} {suite}: cluster AUROC {c['point']} CI[{c['lo']},{c['hi']}] "
                f"len->label AUROC={a['auroc_label_from_answer_length_alone']} "
                f"pair dComp mean={w['composite_delta_hall_minus_correct_mean']} "
                f"frac_hall_lower={w['frac_hall_lower_composite']}"
            )
        out["models"][slug] = {"model_slug": slug, "suites": suites}
    path = REPORTS / "pairing_and_paired_bootstrap_audit.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("Wrote", path)


if __name__ == "__main__":
    main()
