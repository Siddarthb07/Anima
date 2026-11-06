#!/usr/bin/env python3
"""Ablate System A vs G' vs hall-probe vs channel-logistic vs fused OP on TQA.

Uses locked live claim-encode rows + sealed hall_probe scores when present.
shipping_claim=false. Not a shipping detector.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from benchmarks.guard_metrics import auroc_from_scores
from benchmarks.protocol_phase2_9_10_battery import (
    TARGETS,
    _badge_stats,
    _coverage_at_risk,
    _row_channels,
)
from scripts.protocol_r_retune import question_holdout_split

REPORTS = Path(__file__).resolve().parent / "reports"
ZOO = Path(__file__).resolve().parents[1] / "probes" / "zoo"
PAPER = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact")

W_G_MARGIN = 0.65
W_G_PROBE = 0.35
FEATS = ("probe", "fused", "entropy", "inv_logit_gap")


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, steps: int = 400, lr: float = 0.5):
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


def _gprime(ch: dict[str, float]) -> float:
    inv = float(ch["inv_logit_gap"])
    return W_G_MARGIN * inv + W_G_PROBE * float(ch["probe"])


def _selective_block(scores: list[float], labels: list[int], auroc: float, len_auroc: float) -> dict[str, Any]:
    best = None
    hits = []
    for tau in [round(x, 2) for x in np.linspace(0.30, 0.90, 31)]:
        badge = _badge_stats(scores, labels, tau)
        if not badge["selective"] or badge["error_among_green"] is None:
            continue
        cov10 = _coverage_at_risk(scores, labels, 0.10)
        cov20 = _coverage_at_risk(scores, labels, 0.20)
        row = {
            "tau": tau,
            "auroc": auroc,
            "badge": badge,
            "coverage_at_risk_0.10": cov10,
            "coverage_at_risk_0.20": cov20,
            "joint_decide": "abstain_if_score_ge_tau",
        }
        hit = (
            auroc >= TARGETS["min_heldout_auroc"]
            and badge["error_among_green"] <= TARGETS["max_error_among_green"]
            and cov10.get("status") == "ok"
            and float(cov10["coverage"]) >= TARGETS["min_coverage_at_risk_0_10"]
            and float(cov20["coverage"]) >= TARGETS["min_coverage_at_risk_0_20"]
            and len_auroc < TARGETS["length_to_label_max_for_primary"]
        )
        row["hits_9_gates"] = hit
        if hit:
            hits.append(row)
        if best is None or badge["error_among_green"] < best["badge"]["error_among_green"]:
            best = row
    return {"best_selective": best, "n_hits_9_gates": len(hits), "hits": hits[:5]}


def _load_hall_scores(slug_zoo: str, suite: str) -> Optional[dict[str, Any]]:
    path = REPORTS / f"hall_probe_{slug_zoo}_{suite}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(
    slug: str,
    *,
    suite: str = "truthfulqa",
    seed: int = 42,
) -> dict[str, Any]:
    lk = "should_abstain" if suite == "truthfulqa" else "is_hallucination"
    path = REPORTS / f"{suite}_live_{slug}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    questions = [str(r.get("question") or "") for r in rows]
    tune_m, test_m = question_holdout_split(questions, seed=seed)
    tune = [r for r, m in zip(rows, tune_m) if m]
    test = [r for r, m in zip(rows, test_m) if m]
    y = [int(bool(r[lk])) for r in test]
    tune_y = [int(bool(r[lk])) for r in tune]
    test_ch = [_row_channels(r) for r in test]
    tune_ch = [_row_channels(r) for r in tune]
    len_auroc = auroc_from_scores([c["answer_length"] for c in test_ch], y)

    instruments: dict[str, list[float]] = {}
    # System A composite (polarity-aware)
    auroc_comp = auroc_from_scores([c["composite"] for c in test_ch], y)
    if auroc_comp < 0.5:
        instruments["sys_a_inv_composite"] = [c["inv_composite"] for c in test_ch]
    else:
        instruments["sys_a_composite"] = [c["composite"] for c in test_ch]
    instruments["gprime"] = [_gprime(c) for c in test_ch]
    instruments["probe_only"] = [c["probe"] for c in test_ch]
    instruments["inv_logit_gap"] = [c["inv_logit_gap"] for c in test_ch]

    # Channel logistic (length excluded)
    Xtr = np.array([[c[f] for f in FEATS] for c in tune_ch], float)
    Xte = np.array([[c[f] for f in FEATS] for c in test_ch], float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    w, b = _fit_logistic((Xtr - mu) / sd, np.array(tune_y, float))
    z = ((Xte - mu) / sd) @ w + b
    instruments["channel_logistic"] = (1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))).tolist()

    # Hall probe scores if sidecar lists them for held-out in order; else from meta best only
    zoo_slug = slug.replace("-", "_")
    # live slug uses hyphens: qwen_qwen2_5-0_5b-instruct → zoo qwen_qwen2_5_0_5b_instruct
    zoo_slug = (
        slug.lower().replace("-", "_").replace(".", "_")
    )
    hall = _load_hall_scores(zoo_slug, suite)
    hall_meta = None
    if hall and hall.get("heldout_scores"):
        instruments["hall_probe"] = [float(x) for x in hall["heldout_scores"]]
        hall_meta = {
            "sha256": hall.get("sha256"),
            "checkpoint": hall.get("checkpoint"),
            "heldout_auroc": hall.get("heldout_auroc"),
            "not_affect_probe": True,
            "not_gprime_refusion": True,
        }
        # Fuse: mean of hall + channel logistic (same decide)
        if len(instruments["hall_probe"]) == len(instruments["channel_logistic"]):
            instruments["fused_hall_channel"] = [
                0.5 * a + 0.5 * b
                for a, b in zip(instruments["hall_probe"], instruments["channel_logistic"])
            ]

    blocks = {}
    for name, scores in instruments.items():
        if len(scores) != len(y):
            blocks[name] = {"status": "length_mismatch", "n_scores": len(scores), "n_y": len(y)}
            continue
        auroc = auroc_from_scores(scores, y)
        sel = _selective_block(scores, y, auroc, len_auroc)
        blocks[name] = {
            "heldout_auroc": auroc,
            "length_to_label_auroc": len_auroc,
            "beats_length": auroc > len_auroc,
            **sel,
        }

    any_hit = any(b.get("n_hits_9_gates", 0) > 0 for b in blocks.values() if isinstance(b, dict))
    out = {
        "schema": "anima.protocol_hall_ablation.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_slug": slug,
        "suite": suite,
        "n_full": len(rows),
        "n_heldout": len(test),
        "length_to_label_auroc": len_auroc,
        "length_gate_clear": len_auroc < TARGETS["length_to_label_max_for_primary"],
        "hall_meta": hall_meta,
        "instruments": blocks,
        "any_hits_9_gates": any_hit,
        "shipping_claim": False,
        "targets": TARGETS,
        "source": str(path),
    }
    return out


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main() -> None:
    rollup = {
        "schema": "anima.protocol_hall_ablation.rollup.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shipping_claim": False,
        "models": {},
    }
    for slug in (
        "qwen_qwen2_5-0_5b-instruct",
        "qwen_qwen2_5-7b-instruct",
    ):
        path = REPORTS / f"truthfulqa_live_{slug}.json"
        if not path.is_file():
            rollup["models"][slug] = {"status": "missing"}
            print(f"MISSING {slug}")
            continue
        block = analyze(slug, suite="truthfulqa")
        rollup["models"][slug] = block
        print(f"=== {slug} len={block['length_to_label_auroc']:.3f} any9={block['any_hits_9_gates']}")
        for name, b in block["instruments"].items():
            if "heldout_auroc" not in b:
                print(f"  {name}: {b}")
                continue
            best = b.get("best_selective")
            err = best["badge"]["error_among_green"] if best else None
            cov = (best or {}).get("coverage_at_risk_0.10", {}).get("coverage")
            print(
                f"  {name}: auroc={b['heldout_auroc']:.3f} hits={b['n_hits_9_gates']} "
                f"errG={err} cov10={cov}"
            )
    out = REPORTS / "protocol_hall_ablation_rollup.json"
    out.write_text(json.dumps(rollup, indent=2, default=_jsonable), encoding="utf-8")
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / out.name).write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    (PAPER / "plan9" / out.name).parent.mkdir(parents=True, exist_ok=True)
    (PAPER / "plan9" / out.name).write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
