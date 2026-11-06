#!/usr/bin/env python3
"""Multi-guard claim-encode harness under one abstain rule.

Five higher-is-risk scores, same decide(score, tau) = (score >= tau):

  entropy      answer-span mean predictive entropy
  margin       answer-span mean of 1/(1 + top1-top2)  (existing logit_gap)
  composite    evaluate_guard composite (0.55 fused + 0.45 probe)
  hall_probe   late-layer linear head, fit on the tune fold only (TQA primary)
  perplexity   answer-span mean token NLL (monotone with perplexity)

Tau is Youden's J on the question-held-out tune fold, then frozen.
Held-out AUROCs are paired with DeLong and Holm within each cell.
Seeds {42, 43, 44} resample pairs and the question split.

shipping_claim=false. Claim-encode (Q+A), not free generation.
Offline: refuses models that are not already in HF_HOME.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "benchmarks" / "reports"
GUARDS = ("entropy", "margin", "composite", "hall_probe", "perplexity")

# Small cached models that fit a 16GB T4 and a 15GiB host with no swap.
# 7B fp16 does not fit the T4; 3B is omitted so the process is not OOM-killed.
DEFAULT_MODELS = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
)

DECIDE_NAME = "abstain_if_score_ge_tau"


def decide(score: float, tau: float) -> bool:
    """Shared abstain rule for every guard. Higher score = higher risk."""
    return float(score) >= float(tau)


def _norm_sf(z: float) -> float:
    """Upper tail of the standard normal, 1 - Phi(z)."""
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, same order as the input."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (pvals[i], i))
    adj = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        raw = (m - rank) * float(pvals[i])
        running = max(running, raw)
        adj[i] = min(1.0, running)
    return [round(a, 6) for a in adj]


def _midrank(x: np.ndarray) -> np.ndarray:
    """DeLong midranks (1-based), ties get the average rank."""
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = t
    return out


def delong_pair(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    labels: Sequence[int],
) -> dict[str, Any]:
    """Two-sided DeLong test of AUROC(a) - AUROC(b) on the same labels.

    Needs at least two positives and two negatives. Returns p=None when the
    covariance is undefined.
    """
    y = np.asarray(labels, dtype=np.int64)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape or a.shape != y.shape:
        return {"status": "length_mismatch"}
    pos = y == 1
    n1 = int(pos.sum())
    n0 = int((~pos).sum())
    if n1 < 2 or n0 < 2:
        return {"status": "too_few_class", "n_pos": n1, "n_neg": n0}
    order = np.argsort(-y, kind="mergesort")
    preds = np.vstack([a, b])[:, order]
    m = n1
    n = n0
    k = 2
    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for r in range(k):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    contrast = np.array([1.0, -1.0])
    var = float(np.asarray(contrast @ cov @ contrast).reshape(-1)[0])
    delta = float(aucs[0] - aucs[1])
    if not math.isfinite(var) or var <= 1e-18:
        z = 0.0 if abs(delta) < 1e-12 else float("inf")
        p = 1.0 if abs(delta) < 1e-12 else 0.0
        status = "zero_variance"
    else:
        z = delta / math.sqrt(var)
        p = min(1.0, 2.0 * _norm_sf(abs(z)))
        status = "ok"
    return {
        "status": status,
        "auroc_a_delong": round(float(aucs[0]), 4),
        "auroc_b_delong": round(float(aucs[1]), 4),
        "delta": round(delta, 4),
        "var": None if not math.isfinite(var) else round(var, 8),
        "z": None if not math.isfinite(z) else round(z, 4),
        "p": None if not math.isfinite(p) else round(p, 6),
        "n_pos": n1,
        "n_neg": n0,
    }


def delong_holm_family(
    score_map: dict[str, Sequence[float]],
    labels: Sequence[int],
    names: Sequence[str] = GUARDS,
) -> dict[str, Any]:
    """All pairwise DeLong tests among guards, Holm-adjusted inside the cell."""
    present = [n for n in names if n in score_map]
    raw: list[dict[str, Any]] = []
    pvals: list[float] = []
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            rec = delong_pair(score_map[a], score_map[b], labels)
            rec["a"] = a
            rec["b"] = b
            raw.append(rec)
            p = rec.get("p")
            pvals.append(1.0 if p is None else float(p))
    adj = holm(pvals)
    for rec, p_h in zip(raw, adj):
        rec["p_holm"] = p_h
        rec["reject_holm_0.05"] = bool(rec.get("p") is not None and p_h < 0.05)
    n_rej = sum(1 for r in raw if r["reject_holm_0.05"])
    return {
        "family": "all_pairs_within_cell",
        "n_tests": len(raw),
        "n_reject_holm_0.05": n_rej,
        "pairs": raw,
    }


def youden_tau(scores: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """Tune-fold threshold maximizing TPR - FPR. Abstain when score >= tau."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    finite = np.isfinite(s)
    s, y = s[finite], y[finite]
    if len(s) == 0 or len(set(y.tolist())) < 2:
        med = float(np.median(s)) if len(s) else 0.5
        return {"tau": med, "j": None, "status": "degenerate_tune"}
    best_j = -2.0
    best_tau = float(np.median(s))
    for tau in np.unique(s):
        pred = s >= tau
        tp = int(np.sum(pred & (y == 1)))
        fn = int(np.sum(~pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        tn = int(np.sum(~pred & (y == 0)))
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        j = tpr - fpr
        if j > best_j + 1e-12:
            best_j = j
            best_tau = float(tau)
    return {"tau": round(best_tau, 6), "j": round(best_j, 4), "status": "ok"}


def _badge(scores: Sequence[float], labels: Sequence[int], tau: float) -> dict[str, Any]:
    abstain = [decide(s, tau) for s in scores]
    n = len(scores)
    n_abs = sum(abstain)
    n_green = n - n_abs
    n_mis = sum(1 for a, y in zip(abstain, labels) if (not a) and int(y) == 1)
    tp = sum(1 for a, y in zip(abstain, labels) if a and int(y) == 1)
    tn = sum(1 for a, y in zip(abstain, labels) if (not a) and int(y) == 0)
    acc = (tp + tn) / max(1, n)
    return {
        "tau": tau,
        "decide": DECIDE_NAME,
        "n": n,
        "n_abstain": n_abs,
        "abstain_events": f"{n_abs}/{n}",
        "n_green": n_green,
        "error_among_green": round(n_mis / n_green, 4) if n_green else None,
        "false_green_rate": round(n_mis / max(1, n), 4),
        "abstain_accuracy": round(acc, 4),
        "selective": bool(0 < n_abs < n),
        "never_abstain": n_abs == 0,
    }


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        if not math.isfinite(v):
            return None
        return v
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(type(o))


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")
    os.replace(tmp, path)


def _cached(model_name: str) -> bool:
    home = Path(os.environ.get("HF_HOME", str(ROOT / ".hf_cache")))
    slug = "models--" + model_name.replace("/", "--")
    snaps = home / "hub" / slug / "snapshots"
    if not snaps.is_dir():
        return False
    for snap in snaps.iterdir():
        if not snap.is_dir():
            continue
        weights = list(snap.glob("*.safetensors")) or list(snap.glob("*.bin"))
        if (snap / "config.json").is_file() and weights:
            return True
    return False


def _span_start(n: int) -> int:
    return max(0, n - max(4, n // 3))


def _mean(xs: Sequence[float]) -> float:
    vals = [float(v) for v in xs if v == v and math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _answer_nll(rows: list[dict[str, Any]], start: int) -> float:
    """Mean NLL of tokens inside the answer span. Logits at t predict token t+1."""
    import torch

    nlls: list[float] = []
    n = len(rows)
    for t in range(n - 1):
        if t + 1 < start:
            continue
        logits = rows[t].get("logits")
        if logits is None:
            continue
        tid = int(rows[t + 1]["token_id"])
        lt = logits.float()
        if not torch.isfinite(lt).all():
            continue
        logp = torch.log_softmax(lt, dim=-1)[tid]
        val = float(-logp)
        if math.isfinite(val):
            nlls.append(val)
    return _mean(nlls)


def _collect_rows(
    extractor: Any,
    probe: Any,
    pairs: list[tuple[str, str, bool]],
    *,
    model_name: str,
    label_key: str,
) -> list[dict[str, Any]]:
    import torch

    from core.guard import evaluate_guard
    from core.prompt_format import format_qa_claim

    late = extractor.late_layer
    out: list[dict[str, Any]] = []
    for i, (question, answer, label) in enumerate(pairs):
        text = format_qa_claim(
            extractor.tokenizer, question, answer, model_name=model_name
        )
        enc = extractor.encode_sequence(text, max_length=192, apply_chat_template=False)
        if not enc:
            continue
        start = _span_start(len(enc))
        span = enc[start:]
        ent = _mean([float(r["uncertainty_signals"]["entropy"]) for r in span])
        margin = _mean([float(r["uncertainty_signals"]["logit_gap"]) for r in span])
        fused = _mean([float(r["uncertainty_signals"]["fused"]) for r in span])
        attn = _mean([float(r["uncertainty_signals"]["attn_entropy"]) for r in span])
        mean_nll = _answer_nll(enc, start)
        last = enc[-1]
        with torch.no_grad():
            affect_t = probe(last["activations"])
        affect = {
            "valence": float(affect_t["valence"]),
            "arousal": float(affect_t["arousal"]),
            "uncertainty": float(affect_t["uncertainty"]),
        }
        uncertainty_signals = {
            "entropy": None if not math.isfinite(ent) else round(ent, 4),
            "logit_gap": None if not math.isfinite(margin) else round(margin, 4),
            "attn_entropy": None if not math.isfinite(attn) else round(attn, 4),
            "fused": None if not math.isfinite(fused) else round(fused, 4),
        }
        # evaluate_guard expects floats; NaN channels become the guard default.
        us_guard = {
            k: (0.5 if v is None else float(v)) for k, v in uncertainty_signals.items()
        }
        token_text = "".join(r["token_text"] for r in span)
        g = evaluate_guard(
            affect=affect,
            uncertainty_signals=us_guard,
            token_text=token_text,
        )
        h = last["activations"][late]
        hidden = h.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        out.append(
            {
                "question": question[:240],
                "answer": answer[:240],
                label_key: bool(label),
                "entropy": ent,
                "margin": margin,
                "composite": float(g.composite_score),
                "perplexity_nll": mean_nll,
                "perplexity": float(math.exp(min(mean_nll, 20.0)))
                if math.isfinite(mean_nll)
                else float("nan"),
                "production_abstain": bool(g.abstain_recommended),
                "hidden": hidden,
                "answer_length": float(len(answer)),
            }
        )
        del enc
        if (i + 1) % 20 == 0:
            print(f"  encoded {i + 1}/{len(pairs)}", flush=True)
    return out


def _fit_hall_probe(
    hiddens: np.ndarray,
    labels: np.ndarray,
    tune_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    seed: int,
    epochs: int = 40,
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Train on the tune fold only. Orient with train AUROC so higher = risk.

    Returns (test_scores, tune_scores, meta). Tune scores are for tau only.
    """
    import torch
    import torch.nn as nn

    from benchmarks.guard_metrics import auroc_from_scores

    xtr = hiddens[tune_mask]
    ytr = labels[tune_mask].astype(np.float64)
    xte = hiddens[test_mask]
    meta: dict[str, Any] = {
        "fit": "full_tune_fold_fixed_epochs",
        "epochs": epochs,
        "n_train": int(tune_mask.sum()),
        "oriented_on_train_only": True,
        "flipped": False,
    }
    n_te = int(np.asarray(test_mask).sum())
    n_tr = int(np.asarray(tune_mask).sum())
    if len(set(ytr.tolist())) < 2 or len(xtr) < 4:
        meta["status"] = "degenerate_tune"
        return [float("nan")] * n_te, [float("nan")] * n_tr, meta

    torch.manual_seed(seed)
    mu = xtr.mean(0)
    sd = xtr.std(0) + 1e-6
    xtr_n = (xtr - mu) / sd
    xte_n = (xte - mu) / sd

    class _Head(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.fc = nn.Linear(dim, 1)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.fc(h)).squeeze(-1)

    head = _Head(xtr_n.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-3)
    loss_fn = nn.BCELoss()
    xt = torch.tensor(xtr_n, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = head(xt).clamp(1e-6, 1.0 - 1e-6)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        tr_s = head(xt).detach().cpu().numpy().tolist()
        te_s = head(torch.tensor(xte_n, dtype=torch.float32)).detach().cpu().numpy().tolist()
    train_auroc = auroc_from_scores(tr_s, [int(v) for v in ytr.tolist()])
    meta["train_auroc_before_orient"] = train_auroc
    if train_auroc < 0.5:
        te_s = [1.0 - float(s) for s in te_s]
        tr_s = [1.0 - float(s) for s in tr_s]
        meta["flipped"] = True
    meta["status"] = "ok"
    meta["hidden_dim"] = int(xtr_n.shape[1])
    return [float(s) for s in te_s], [float(s) for s in tr_s], meta


def _eval_cell(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    seed: int,
    ci_B: int,
) -> dict[str, Any]:
    from benchmarks.guard_metrics import auroc_from_scores, bootstrap_auroc_ci
    from scripts.protocol_r_retune import question_holdout_split

    questions = [str(r["question"]) for r in rows]
    tune_m, test_m = question_holdout_split(questions, seed=seed)
    labels = np.array([int(bool(r[label_key])) for r in rows], dtype=np.int64)
    hiddens = np.stack([r["hidden"] for r in rows])

    def col(key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in rows], dtype=np.float64)

    entropy = col("entropy")
    margin = col("margin")
    composite = col("composite")
    nll = col("perplexity_nll")
    # Drop rows that are non-finite on a non-learned guard so pairs stay aligned.
    base_ok = (
        np.isfinite(entropy)
        & np.isfinite(margin)
        & np.isfinite(composite)
        & np.isfinite(nll)
    )
    tune_m = np.asarray(tune_m) & base_ok
    test_m = np.asarray(test_m) & base_ok
    n_drop = int((~base_ok).sum())
    hall_te, hall_tr, hall_meta = _fit_hall_probe(
        hiddens, labels, tune_m, test_m, seed=seed
    )

    y_te = labels[test_m].tolist()
    y_tr = labels[tune_m].tolist()
    raw_test = {
        "entropy": entropy[test_m].tolist(),
        "margin": margin[test_m].tolist(),
        "composite": composite[test_m].tolist(),
        "perplexity": nll[test_m].tolist(),
        "hall_probe": hall_te,
    }
    raw_tune = {
        "entropy": entropy[tune_m].tolist(),
        "margin": margin[tune_m].tolist(),
        "composite": composite[tune_m].tolist(),
        "perplexity": nll[tune_m].tolist(),
        "hall_probe": hall_tr,
    }

    definitions = {
        "entropy": "answer_span_mean_predictive_entropy",
        "margin": "answer_span_mean_1_over_1_plus_top1_minus_top2",
        "composite": "evaluate_guard_0.55_fused_plus_0.45_probe",
        "hall_probe": "late_layer_linear_sigmoid_tune_fold_only",
        "perplexity": "answer_span_mean_token_nll",
    }
    guards: dict[str, Any] = {}
    test_scores: dict[str, list[float]] = {}
    for name in GUARDS:
        if name == "hall_probe" and hall_meta.get("status") != "ok":
            guards[name] = {"status": hall_meta.get("status", "missing"), "meta": hall_meta}
            continue
        tau_info = youden_tau(raw_tune[name], y_tr)
        scores = raw_test[name]
        if any(not math.isfinite(float(s)) for s in scores):
            guards[name] = {"status": "nonfinite_test", "definition": definitions[name]}
            continue
        tau = float(tau_info["tau"])
        auroc = auroc_from_scores(scores, y_te)
        guards[name] = {
            "status": "ok",
            "definition": definitions[name],
            "higher_is_risk": True,
            "tau_youden_tune": tau_info,
            "auroc_heldout": auroc,
            "auroc_ci95": bootstrap_auroc_ci(scores, y_te, B=ci_B, seed=seed),
            "badge_heldout": _badge(scores, y_te, tau),
        }
        test_scores[name] = scores

    if "hall_probe" in guards and guards["hall_probe"].get("status") == "ok":
        guards["hall_probe"]["meta"] = hall_meta

    length_scores = [float(r["answer_length"]) for r, m in zip(rows, test_m) if m]
    length_auroc = (
        auroc_from_scores(length_scores, y_te) if len(length_scores) == len(y_te) else None
    )
    family = delong_holm_family(test_scores, y_te, names=[n for n in GUARDS if n in test_scores])
    n_prod = sum(int(r["production_abstain"]) for r, m in zip(rows, test_m) if m)
    return {
        "status": "ok",
        "n_encoded": len(rows),
        "n_dropped_nonfinite": n_drop,
        "n_tune": int(np.asarray(tune_m).sum()),
        "n_test": int(np.asarray(test_m).sum()),
        "n_pos_test": int(sum(y_te)),
        "n_neg_test": int(len(y_te) - sum(y_te)),
        "pilot_not_confirmatory": bool(len(y_te) < 100),
        "length_to_label_auroc_heldout": length_auroc,
        "production_evaluate_guard_abstain_on_test": f"{n_prod}/{len(y_te)}",
        "guards": guards,
        "delong_holm": family,
        "heldout_labels": y_te,
        "heldout_scores": {k: [round(float(s), 6) for s in v] for k, v in test_scores.items()},
    }


def _release(extractor: Any) -> None:
    import torch

    try:
        extractor.cleanup()
    except Exception:
        pass
    model = getattr(extractor, "model", None)
    if model is not None:
        try:
            del extractor.model
        except Exception:
            pass
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", str(ROOT / ".hf_cache"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["ANIMA_FORCE_CPU"] = "0" if str(args.device).startswith("cuda") else "1"
    os.environ["ANIMA_OUTPUT_ATTENTIONS"] = "0"
    os.environ["ANIMA_LOAD_8BIT"] = "0"
    os.environ["ANIMA_PREFER_TEXT_PROBE"] = "1"

    import torch

    from benchmarks.live_guard_eval import load_halueval_pairs, load_truthfulqa_pairs
    from core.extractor import ActivationExtractor
    from probes.linear_probe import AffectProbe
    from probes.zoo_io import load_probe_into

    models = list(args.models) if args.models else list(DEFAULT_MODELS)
    kept: list[str] = []
    skipped: list[dict[str, str]] = []
    for name in models:
        if _cached(name):
            kept.append(name)
        else:
            skipped.append({"model": name, "reason": "not_in_hf_cache"})
            print(f"SKIP uncached {name}", flush=True)
    if not kept:
        raise SystemExit("No requested models are present in the HF cache. Refusing to download.")

    suites: list[tuple[str, str, Callable[..., list]]] = []
    if args.benchmark in ("truthfulqa", "both"):
        suites.append(("truthfulqa", "should_abstain", load_truthfulqa_pairs))
    if args.benchmark in ("halueval", "both"):
        suites.append(("halueval", "is_hallucination", load_halueval_pairs))

    device_name = None
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    rollup: dict[str, Any] = {
        "schema": "anima.multi_guard_claim_encode.v1",
        "shipping_claim": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "decide": DECIDE_NAME,
        "guards": list(GUARDS),
        "seeds": list(args.seeds),
        "max_pairs": args.max_pairs,
        "benchmark": args.benchmark,
        "device": args.device,
        "device_name": device_name,
        "attentions": os.environ.get("ANIMA_OUTPUT_ATTENTIONS"),
        "models_requested": models,
        "models_ran": kept,
        "models_skipped": skipped,
        "cells": [],
        "honesty": {
            "shipping_claim": False,
            "claim_encode_not_free_generation": True,
            "no_new_weight_downloads": True,
            "same_decide_for_every_guard": DECIDE_NAME,
            "tau_fit_on_tune_fold_only": True,
            "hall_probe_not_affect_probe": True,
            "hall_probe_primary_suite": "truthfulqa",
            "composite_not_silently_inverted": True,
            "perplexity_score_is_mean_nll": True,
            "margin_score_is_uncertainty_coded_gap": True,
            "attentions_off_fused_renormalizes": os.environ.get("ANIMA_OUTPUT_ATTENTIONS") == "0",
            "pilot_unless_n_test_ge_100": True,
            "confirmatory_n_target": 300,
            "delong_holm_family": "all_pairs_within_cell",
            "seven_b_not_run": "T4 16GB and 15GiB host RAM, no swap; 7B fp16 does not fit",
        },
    }
    out = Path(args.out)
    _atomic_write(out, rollup)
    print(f"wrote heartbeat {out}", flush=True)

    for model_name in kept:
        print(f"\n===== {model_name} =====", flush=True)
        extractor = None
        try:
            extractor = ActivationExtractor(model_name, device=args.device)
            probe = AffectProbe(extractor.hidden_dim, len(extractor.layer_indices))
            probe_meta = load_probe_into(probe, model_name, map_location="cpu")
            probe_origin = probe_meta.get("probe_origin")
            print(f"probe_origin={probe_origin} hidden={extractor.hidden_dim}", flush=True)
            for suite, label_key, loader in suites:
                for seed in args.seeds:
                    cell_id = f"{model_name} {suite} seed={seed}"
                    print(f"\n--- {cell_id} ---", flush=True)
                    cell: dict[str, Any] = {
                        "model": model_name,
                        "suite": suite,
                        "seed": int(seed),
                        "label_key": label_key,
                        "probe_origin": probe_origin,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    }
                    try:
                        pairs = loader(max_pairs=args.max_pairs, seed=int(seed))
                        print(f"  pairs={len(pairs)}", flush=True)
                        encoded = _collect_rows(
                            extractor,
                            probe,
                            pairs,
                            model_name=model_name,
                            label_key=label_key,
                        )
                        stats = _eval_cell(
                            encoded,
                            label_key=label_key,
                            seed=int(seed),
                            ci_B=args.ci_B,
                        )
                        cell.update(stats)
                        brief = {
                            g: (cell["guards"].get(g) or {}).get("auroc_heldout")
                            for g in GUARDS
                        }
                        print(
                            f"CELL {cell_id} n_test={cell.get('n_test')} auroc={brief}",
                            flush=True,
                        )
                    except Exception as e:
                        cell["status"] = "failed"
                        cell["error"] = str(e)
                        cell["traceback"] = traceback.format_exc()[-3000:]
                        print(f"CELL FAILED {cell_id}: {e}", flush=True)
                    cell["finished_at"] = datetime.now(timezone.utc).isoformat()
                    rollup["cells"].append(cell)
                    rollup["updated_at"] = cell["finished_at"]
                    _atomic_write(out, rollup)
                    gc.collect()
        except Exception as e:
            print(f"MODEL FAILED {model_name}: {e}", flush=True)
            rollup["cells"].append(
                {
                    "model": model_name,
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc()[-3000:],
                }
            )
            _atomic_write(out, rollup)
        finally:
            if extractor is not None:
                _release(extractor)
                extractor = None

    rollup["status"] = "complete"
    rollup["finished_at"] = datetime.now(timezone.utc).isoformat()
    rollup["n_cells"] = len(rollup["cells"])
    rollup["n_ok"] = sum(1 for c in rollup["cells"] if c.get("status") == "ok")
    _atomic_write(out, rollup)
    print(f"DONE {out} ok={rollup['n_ok']}/{rollup['n_cells']}", flush=True)
    return rollup


def _self_test() -> None:
    # Holm: p = 0.01, 0.04, 0.03 → adjusted 0.03, 0.06, 0.06
    adj = holm([0.01, 0.04, 0.03])
    assert adj == [0.03, 0.06, 0.06], adj
    assert decide(0.8, 0.8) is True
    assert decide(0.79, 0.8) is False

    rng = np.random.default_rng(0)
    y = np.array([1] * 40 + [0] * 40)
    signal = y + rng.normal(0, 0.35, size=80)
    noise = rng.normal(0, 1, size=80)
    same = delong_pair(signal, signal, y)
    assert same["status"] in ("ok", "zero_variance")
    assert same["p"] is not None and same["p"] > 0.5, same
    diff = delong_pair(signal, noise, y)
    assert diff["status"] == "ok", diff
    assert diff["p"] is not None and diff["p"] < 0.05, diff
    assert diff["auroc_a_delong"] > diff["auroc_b_delong"]

    tau = youden_tau(signal.tolist(), y.tolist())
    assert tau["status"] == "ok" and tau["j"] is not None and tau["j"] > 0.2

    family = delong_holm_family(
        {"entropy": signal, "margin": noise, "composite": signal * 0.5},
        y,
        names=("entropy", "margin", "composite"),
    )
    assert family["n_tests"] == 3
    assert all("p_holm" in p for p in family["pairs"])
    print("self-test ok", flush=True)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Multi-guard claim-encode (offline cached LMs)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max-pairs", type=int, default=160)
    ap.add_argument("--benchmark", choices=("truthfulqa", "halueval", "both"), default="both")
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    ap.add_argument("--ci-B", type=int, default=2000)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPORTS / "multi_guard_claim_encode_rollup.json",
    )
    args = ap.parse_args(argv)
    if args.self_test:
        _self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
