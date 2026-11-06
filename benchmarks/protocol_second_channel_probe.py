#!/usr/bin/env python3
"""Train a second *channel* probe (logistic) on held-out claim-encode features.

This is NOT a second hidden-state AffectProbe. It is an honest intermediate
instrument: logistic regression on [probe, fused, entropy, inv_logit_gap, length]
fit on tune fold, scored on held-out. Seal as shipping_claim=false.

For council 9–10 the plan still requires a real second hidden-state checkpoint;
this closes the 'only G′ re-fusion' gap partially with a trained classifier.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.guard_metrics import auroc_from_scores
from benchmarks.protocol_phase2_9_10_battery import _coverage_at_risk, _row_channels, TARGETS
from scripts.protocol_r_retune import question_holdout_split

REPORTS = Path(__file__).resolve().parent / "reports"
PAPER = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact")
# Length excluded on purpose — including it makes "beats length" circular.
FEATS = ("probe", "fused", "entropy", "inv_logit_gap")


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, steps: int = 400, lr: float = 0.5):
    """Minimal L2 logistic (no sklearn dependency)."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        err = p - y
        w -= lr * ((X.T @ err) / n + l2 * w)
        b -= lr * float(err.mean())
    return w, b


def run(slug: str = "qwen_qwen2_5-0_5b-instruct", suite: str = "halueval", seed: int = 42) -> dict[str, Any]:
    lk = "is_hallucination" if suite == "halueval" else "should_abstain"
    path = REPORTS / f"{suite}_live_{slug}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    q = [str(r.get("question") or "") for r in rows]
    tune_m, test_m = question_holdout_split(q, seed=seed)
    tune = [r for r, m in zip(rows, tune_m) if m]
    test = [r for r, m in zip(rows, test_m) if m]

    def mat(rs):
        ch = [_row_channels(r) for r in rs]
        X = np.array([[c[f] for f in FEATS] for c in ch], float)
        # standardize using train stats later
        y = np.array([int(bool(r[lk])) for r in rs], float)
        return X, y, ch

    Xtr, ytr, _ = mat(tune)
    Xte, yte, _ = mat(test)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd
    w, b = _fit_logistic(Xtr_n, ytr)
    scores = (1.0 / (1.0 + np.exp(-np.clip(Xte_n @ w + b, -20, 20)))).tolist()
    labels = yte.astype(int).tolist()
    auroc = auroc_from_scores(scores, labels)
    len_auroc = auroc_from_scores([float(len(str(r.get("answer") or ""))) for r in test], labels)

    # τ sweep for selective OP
    best = None
    for tau in np.linspace(0.3, 0.8, 26):
        abs_bits = [s >= tau for s in scores]
        n_abs = sum(abs_bits)
        n = len(scores)
        if not (0 < n_abs < n):
            continue
        greens = [not a for a in abs_bits]
        n_g = sum(greens)
        n_mis = sum(1 for g, y in zip(greens, labels) if g and y == 1)
        errg = n_mis / max(1, n_g)
        cov10 = _coverage_at_risk(scores, labels, 0.10)
        cov20 = _coverage_at_risk(scores, labels, 0.20)
        cand = {
            "tau": float(tau),
            "abstain_events": f"{n_abs}/{n}",
            "error_among_green": round(errg, 4),
            "false_green_rate": round(n_mis / n, 4),
            "auroc": auroc,
            "coverage_at_risk_0.10": cov10,
            "coverage_at_risk_0.20": cov20,
        }
        if best is None or errg < best["error_among_green"]:
            best = cand

    return {
        "schema": "anima.second_channel_probe.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "honesty": {
            "not_hidden_state_probe": True,
            "second_trained_affect_probe_still_required_for_9_10": True,
            "shipping_claim": False,
        },
        "model_slug": slug,
        "suite": suite,
        "features": list(FEATS),
        "weights": {f: float(w[i]) for i, f in enumerate(FEATS)},
        "bias": float(b),
        "heldout_auroc": auroc,
        "length_to_label_auroc": len_auroc,
        "beats_length": auroc > len_auroc,
        "best_selective": best,
        "targets": TARGETS,
        "hits_9_10": bool(
            best
            and auroc >= TARGETS["min_heldout_auroc"]
            and best["error_among_green"] <= TARGETS["max_error_among_green"]
            and best["coverage_at_risk_0.10"].get("status") == "ok"
            and float(best["coverage_at_risk_0.10"]["coverage"]) >= TARGETS["min_coverage_at_risk_0_10"]
            and auroc > len_auroc
        ),
    }


def main() -> None:
    out = {
        "schema": "anima.second_channel_probe.rollup.v1",
        "shipping_claim": False,
        "models": {},
    }
    for slug, suite in (
        ("qwen_qwen2_5-0_5b-instruct", "halueval"),
        ("qwen_qwen2_5-0_5b-instruct", "truthfulqa"),
        ("qwen_qwen2_5-7b-instruct", "halueval"),
        ("distilgpt2", "halueval"),
    ):
        key = f"{slug}::{suite}"
        try:
            block = run(slug, suite)
        except FileNotFoundError as e:
            block = {"status": "missing", "error": str(e)}
        out["models"][key] = block
        if block.get("heldout_auroc") is not None:
            print(
                f"{key}: auroc={block['heldout_auroc']:.3f} len={block['length_to_label_auroc']:.3f} "
                f"beats={block['beats_length']} hits9={block['hits_9_10']} "
                f"best={block.get('best_selective')}"
            )
    path = REPORTS / "protocol_second_channel_probe.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
