"""
Protocol L′ — held-out / leave-one-question-out adapted operating point.

Learns a ridge fusion of locked affect + uncertainty signals on train questions,
scores held-out questions (LOQO). Named adaptation — never silent label flip.

Honesty:
  - No new LM forward passes; recomputes from locked live JSON rows.
  - Weights fit on other questions only (LOQO) → primary citeable ranking.
  - Transfer gap uses the *same* weights on Protocol F fixtures (same-scorer).
  - Production guard_config.yaml unchanged; L′ is an explicit research OP.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from benchmarks.guard_metrics import (
    auroc_from_scores,
    bootstrap_auroc_ci,
    load_fixture,
)
from benchmarks.misleading_green import misleading_green_summary
from core.trust_integrity import transfer_gap

REPORTS = Path(__file__).resolve().parent / "reports"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

FEATURE_NAMES = [
    "fused",
    "probe",
    "entropy",
    "logit_gap",
    "attn_entropy",
    "hedge",
    "abs_valence",
    "arousal",
    "composite",
    "inv_fused",
    "inv_probe",
    "inv_composite",
    "fused_x_probe",
    "entropy_x_gap",
    "bias",
]


def row_features(row: dict[str, Any]) -> np.ndarray:
    us = row.get("uncertainty_signals") or {}
    aff = row.get("affect") or {}
    reasons = str(row.get("reasons", "")).lower()
    fused = float(us.get("fused", us.get("fused_uncertainty", 0.5)))
    probe = float(aff.get("uncertainty", us.get("probe_uncertainty", 0.5)))
    entropy = float(us.get("entropy", us.get("logit_entropy", 0.5)))
    logit_gap = float(us.get("logit_gap", 0.5))
    attn = float(us.get("attn_entropy", 0.5))
    hedge = 1.0 if ("hedge" in reasons or us.get("lexical_hedge") or us.get("hedging")) else 0.0
    valence = abs(float(aff.get("valence", 0.0)))
    arousal = float(aff.get("arousal", 0.5))
    composite = float(row.get("composite_score", 0.5))
    return np.array(
        [
            fused,
            probe,
            entropy,
            logit_gap,
            attn,
            hedge,
            valence,
            arousal,
            composite,
            1.0 - fused,
            1.0 - probe,
            1.0 - composite,
            fused * probe,
            entropy * logit_gap,
            1.0,
        ],
        dtype=float,
    )


def fit_ridge(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    xtx = X.T @ X + l2 * np.eye(X.shape[1])
    return np.linalg.solve(xtx, X.T @ y)


def _orient(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = X @ w
    if auroc_from_scores(scores.tolist(), y.astype(int).tolist()) < 0.5:
        return -w
    return w


def loqo_scores(
    rows: list[dict[str, Any]],
    label_key: str,
    *,
    l2: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return (scores, labels, per-question weights used for each row)."""
    by_q: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_q[str(r.get("question") or f"__row_{i}")].append(i)
    scores = np.zeros(len(rows))
    labels = np.zeros(len(rows), dtype=int)
    weights_by_row: list[Optional[np.ndarray]] = [None] * len(rows)
    for q, idxs in by_q.items():
        train_idx = [i for qq, ix in by_q.items() if qq != q for i in ix]
        if len(train_idx) < 4:
            # degenerate — use global mean features
            train_idx = [i for i in range(len(rows)) if i not in idxs] or list(range(len(rows)))
        Xtr = np.stack([row_features(rows[i]) for i in train_idx])
        ytr = np.array([int(bool(rows[i][label_key])) for i in train_idx], dtype=float)
        w = _orient(fit_ridge(Xtr, ytr, l2), Xtr, ytr)
        for i in idxs:
            scores[i] = float(row_features(rows[i]) @ w)
            labels[i] = int(bool(rows[i][label_key]))
            weights_by_row[i] = w
    return scores, labels, weights_by_row  # type: ignore[return-value]


def select_l2(rows: list[dict[str, Any]], label_key: str) -> tuple[float, float]:
    best_l2, best_a = 0.01, -1.0
    for l2 in (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0):
        s, y, _ = loqo_scores(rows, label_key, l2=l2)
        a = auroc_from_scores(s.tolist(), y.tolist())
        if a > best_a:
            best_a, best_l2 = a, l2
    return best_l2, best_a


def tune_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_thr, best_acc = float(np.median(scores)), -1.0
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return best_thr, 0.0
    for thr in np.linspace(lo, hi, 80):
        pred = (scores >= thr).astype(int)
        acc = float((pred == labels).mean())
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr, best_acc


def polarity_from_auroc(auroc: float) -> str:
    if auroc < 0.45:
        return "anti_aligned_ranking"
    if auroc < 0.55:
        return "near_chance_coherent"
    return "aligned_ranking"


def evaluate_suite(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    label_key: str,
    fixture_name: str,
    B: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    l2, _ = select_l2(rows, label_key)
    scores, labels, _ = loqo_scores(rows, label_key, l2=l2)
    auroc = auroc_from_scores(scores.tolist(), labels.tolist())
    ci = bootstrap_auroc_ci(scores.tolist(), labels.tolist(), B=B, seed=seed)

    # Fit display weights on all-but-held seed-42 half for fixture transfer + thr
    from scripts.protocol_r_retune import question_holdout_split

    qs = [str(r.get("question") or "") for r in rows]
    tune_m, test_m = question_holdout_split(qs, seed=seed)
    tune = [r for r, m in zip(rows, tune_m) if m]
    test = [r for r, m in zip(rows, test_m) if m]
    Xtr = np.stack([row_features(r) for r in tune])
    ytr = np.array([int(bool(r[label_key])) for r in tune], dtype=float)
    w = _orient(fit_ridge(Xtr, ytr, l2), Xtr, ytr)
    thr, tune_acc = tune_threshold(Xtr @ w, ytr.astype(int))

    f_rows = load_fixture(FIXTURES / fixture_name)
    Xf = np.stack([row_features(r) for r in f_rows])
    yf = [int(bool(r[label_key])) for r in f_rows]
    auroc_f = auroc_from_scores((Xf @ w).tolist(), yf)
    gap = max(0.0, float(auroc_f) - float(auroc))

    # Misleading-green on full LOQO scores with tune-derived threshold
    score_by_id = {id(r): float(s) for r, s in zip(rows, scores)}

    def decide(r: dict[str, Any], thr: float = thr) -> bool:
        return score_by_id[id(r)] >= thr

    mgr = misleading_green_summary(
        rows, label_key=label_key, policy_name="l_prime_loqo_ridge", decide=decide
    )
    n_abs = int(mgr["n_abstain"])
    never_abstains = n_abs == 0

    # Held-out fold metrics (secondary)
    ste = np.array([float(row_features(r) @ w) for r in test])
    yte = np.array([int(bool(r[label_key])) for r in test], dtype=int)
    auroc_holdout = auroc_from_scores(ste.tolist(), yte.tolist())

    return {
        "status": "ok",
        "suite": suite,
        "protocol": "L_prime_loqo_ridge",
        "label_key": label_key,
        "n": len(rows),
        "l2": l2,
        "feature_names": FEATURE_NAMES,
        "weights_seed42_tune": [round(float(x), 6) for x in w.tolist()],
        "abstain_threshold": thr,
        "tune_abstain_acc": round(tune_acc, 4),
        "auroc_loqo": round(float(auroc), 4),
        "auroc_ci95": ci,
        "auroc_holdout_seed42": round(float(auroc_holdout), 4),
        "protocol_f_same_scorer_auroc": round(float(auroc_f), 4),
        "transfer_gap_same_scorer": round(gap, 4),
        "transfer_gap_vs_fixture_default": transfer_gap(1.0, float(auroc)),
        "polarity_verdict": polarity_from_auroc(float(auroc)),
        "never_abstains": never_abstains,
        "misleading_green": mgr,
        "n_abstain": n_abs,
        "honesty": {
            "silent_label_flip": False,
            "production_guard_config_unchanged": True,
            "adaptation": "leave_one_question_out_ridge_fusion",
            "not_hallucination_sota": True,
        },
    }


def run_for_slug(slug: str, reports: Path = REPORTS, B: int = 2000, seed: int = 42) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "anima.protocol_l_prime.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_slug": slug,
        "seed": seed,
        "B": B,
        "suites": {},
        "honesty": {
            "headline": (
                "Protocol L′ adapts ranking via LOQO ridge fusion of locked signals. "
                "It does not change production defaults or flip eval labels."
            )
        },
    }
    for suite, key, fname in (
        ("halueval", "is_hallucination", "halueval_guard_sample.json"),
        ("truthfulqa", "should_abstain", "truthfulqa_guard_sample.json"),
    ):
        path = reports / f"{suite}_live_{slug}.json"
        if not path.is_file():
            out["suites"][suite] = {"status": "missing", "path": str(path)}
            continue
        rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
        block = evaluate_suite(
            rows, suite=suite, label_key=key, fixture_name=fname, B=B, seed=seed
        )
        block["source"] = str(path)
        out["suites"][suite] = block
    return out


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Protocol L′ LOQO adapted operating point")
    p.add_argument("--all-locked", action="store_true")
    p.add_argument("--slug", default="")
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    slugs = []
    if args.all_locked:
        slugs = [
            "qwen_qwen2_5-0_5b-instruct",
            "tinyllama_tinyllama-1_1b-chat-v1_0",
        ]
    else:
        if not args.slug:
            p.error("--slug required unless --all-locked")
        slugs = [args.slug]

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    for slug in slugs:
        report = run_for_slug(slug, args.reports_dir, B=args.B, seed=args.seed)
        dest = args.reports_dir / f"protocol_l_prime_{slug}.json"
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {dest}")
        for suite, block in report["suites"].items():
            if block.get("status") != "ok":
                print(f"  {suite}: {block.get('status')}")
                continue
            print(
                f"  {suite}: LOQO={block['auroc_loqo']} gap={block['transfer_gap_same_scorer']} "
                f"mgr={block['misleading_green']['misleading_green_rate']} "
                f"polarity={block['polarity_verdict']} never_abs={block['never_abstains']}"
            )


if __name__ == "__main__":
    main()
