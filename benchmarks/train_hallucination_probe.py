#!/usr/bin/env python3
"""Train a second *hidden-state* hallucination probe (linear head on late layer).

Tune-fold labels only (question holdout seed 42). Saves zoo checkpoint:
  probes/zoo/{slug}_hall_probe.pt + .meta.json

Honesty: shipping_claim=false; not AffectProbe; not G′ re-fusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from benchmarks.guard_metrics import auroc_from_scores
from benchmarks.live_guard_eval import load_halueval_pairs, load_truthfulqa_pairs
from benchmarks.protocol_phase2_9_10_battery import TARGETS, _badge_stats, _coverage_at_risk
from core.extractor import ActivationExtractor
from core.prompt_format import format_qa_claim
from scripts.protocol_r_retune import question_holdout_split

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "probes" / "zoo"
REPORTS = ROOT / "benchmarks" / "reports"
# Prefer paper artifact dir when present (Windows); else repo-local on GPU hosts.
_PAPER_WIN = Path(r"C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact")
PAPER = _PAPER_WIN if _PAPER_WIN.parent.is_dir() else (ROOT / "artifact" / "prereg")


class HallProbe(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(h)).squeeze(-1)


def _slug(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def _collect_hiddens(
    model_name: str,
    pairs: list[tuple[str, str, bool]],
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    extractor = ActivationExtractor(model_name, device=device)
    late = extractor.late_layer
    xs: list[np.ndarray] = []
    ys: list[int] = []
    qs: list[str] = []
    for i, (q, a, y) in enumerate(pairs):
        text = format_qa_claim(extractor.tokenizer, q, a, model_name=model_name)
        rows = extractor.encode_sequence(text, max_length=256, apply_chat_template=False)
        if not rows:
            continue
        # encode_sequence returns list of token rows with activations
        # Use last token late-layer activation
        last = rows[-1]
        acts = last.get("activations") or last.get("layer_activations") or {}
        if isinstance(acts, dict) and late in acts:
            h = acts[late]
        elif isinstance(acts, dict) and str(late) in acts:
            h = acts[str(late)]
        else:
            # fallback: mean of available
            tensors = [v for v in acts.values() if hasattr(v, "detach")]
            if not tensors:
                continue
            h = tensors[-1]
        if torch.is_tensor(h):
            vec = h.detach().float().cpu().numpy().reshape(-1)
        else:
            vec = np.asarray(h, dtype=np.float32).reshape(-1)
        xs.append(vec)
        ys.append(int(bool(y)))
        qs.append(q)
        if (i + 1) % 10 == 0:
            print(f"  extracted {i+1}/{len(pairs)}", flush=True)
    extractor.cleanup()
    return np.stack(xs), np.asarray(ys, dtype=np.int64), qs


def _train(X: np.ndarray, y: np.ndarray, *, epochs: int = 40, lr: float = 1e-2) -> HallProbe:
    dim = X.shape[1]
    probe = HallProbe(dim)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCELoss()
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    probe.train()
    for ep in range(epochs):
        opt.zero_grad()
        pred = probe(xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1} loss={float(loss):.4f}", flush=True)
    return probe


def _eval_scores(probe: HallProbe, X: np.ndarray) -> list[float]:
    probe.eval()
    with torch.no_grad():
        s = probe(torch.tensor(X, dtype=torch.float32)).cpu().numpy().tolist()
    return [float(v) for v in s]


def run(
    *,
    model_name: str,
    suite: str = "halueval",
    max_pairs: int = 80,
    seed: int = 42,
    device: str = "cpu",
    epochs: int = 40,
) -> dict[str, Any]:
    if suite == "halueval":
        pairs = load_halueval_pairs(max_pairs=max_pairs, seed=seed)
        # labels: is_hallucination
        labeled = [(q, a, bool(y)) for q, a, y in pairs]
    else:
        pairs = load_truthfulqa_pairs(max_pairs=max_pairs, seed=seed)
        labeled = [(q, a, bool(y)) for q, a, y in pairs]

    print(f"Collecting hiddens for {len(labeled)} pairs on {model_name}…", flush=True)
    X, y, questions = _collect_hiddens(model_name, labeled, device=device)
    # Proper train / val / test: first hold out test (50%), then split train/val (80/20 of remainder)
    tune_m, test_m = question_holdout_split(questions, seed=seed)
    Xte, yte = X[np.array(test_m)], y[np.array(test_m)]
    q_tune = [q for q, m in zip(questions, tune_m) if m]
    X_rest, y_rest = X[np.array(tune_m)], y[np.array(tune_m)]
    tr_m, va_m = question_holdout_split(q_tune, seed=seed + 1)
    # holdout_split returns (tune, test); map tune→train, test→val
    Xtr, ytr = X_rest[np.array(tr_m)], y_rest[np.array(tr_m)]
    Xva, yva = X_rest[np.array(va_m)], y_rest[np.array(va_m)]
    print(
        f"train={len(ytr)} val={len(yva)} test={len(yte)} dim={X.shape[1]}",
        flush=True,
    )

    # standardize on train only
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n = (Xtr - mu) / sd
    Xva_n = (Xva - mu) / sd
    Xte_n = (Xte - mu) / sd

    probe = _train(Xtr_n, ytr, epochs=epochs)
    val_scores = _eval_scores(probe, Xva_n)
    val_auroc = auroc_from_scores(val_scores, yva.tolist())
    print(f"val_auroc={val_auroc:.4f}", flush=True)
    scores = _eval_scores(probe, Xte_n)
    labels = yte.tolist()
    auroc = auroc_from_scores(scores, labels)
    # Answer lengths parallel to successful X rows (same order as questions)
    ans_lens = []
    qi = 0
    for q, a, yy in labeled:
        if qi >= len(questions):
            break
        if questions[qi] == q:
            ans_lens.append(float(len(a)))
            qi += 1
    if len(ans_lens) != len(questions):
        ans_lens = [0.0] * len(questions)
    test_lens = [ans_lens[i] for i, m in enumerate(test_m) if m]
    length_auroc = (
        auroc_from_scores(test_lens, labels) if len(test_lens) == len(labels) else float("nan")
    )

    best = None
    for tau in np.linspace(0.3, 0.8, 26):
        badge = _badge_stats(scores, labels, float(tau))
        if not badge["selective"]:
            continue
        cov10 = _coverage_at_risk(scores, labels, 0.10)
        cov20 = _coverage_at_risk(scores, labels, 0.20)
        cand = {
            "tau": float(tau),
            "badge": badge,
            "coverage_at_risk_0.10": cov10,
            "coverage_at_risk_0.20": cov20,
            "auroc": auroc,
        }
        err = badge["error_among_green"]
        if err is None:
            continue
        if best is None or err < best["badge"]["error_among_green"]:
            best = cand

    slug = _slug(model_name)
    ZOO.mkdir(parents=True, exist_ok=True)
    # Suite-specific zoo id so HaluEval train does not clobber TruthfulQA.
    ckpt = ZOO / f"{slug}_hall_probe_{suite}.pt"
    meta_path = ZOO / f"{slug}_hall_probe_{suite}.meta.json"
    payload = {
        "state_dict": probe.state_dict(),
        "mu": mu,
        "sd": sd,
        "hidden_dim": int(X.shape[1]),
        "model_name": model_name,
        "suite_train": suite,
        "seed": seed,
        "late_layer": "extractor.late_layer",
    }
    torch.save(payload, ckpt)
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    meta = {
        "schema": "anima.hallucination_hidden_probe.v1",
        "checkpoint": str(ckpt),
        "sha256": sha,
        "model_name": model_name,
        "train_suite": suite,
        "max_pairs": max_pairs,
        "seed": seed,
        "epochs": epochs,
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_test": int(len(yte)),
        "n_tune": int(len(ytr)),  # legacy alias
        "n_heldout": int(len(yte)),
        "val_auroc": val_auroc,
        "heldout_auroc": auroc,
        "test_auroc": auroc,
        "length_to_label_auroc": length_auroc,
        "split": "question_clustered_train_val_test",
        "pilot_if_n_test_lt_100": bool(len(yte) < 100),
        "heldout_scores": scores,
        "heldout_labels": labels,
        "best_selective": best,
        "shipping_claim": False,
        "not_affect_probe": True,
        "not_gprime_refusion": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets": TARGETS,
        "hits_9_gates": bool(
            best
            and auroc >= TARGETS["min_heldout_auroc"]
            and best["badge"]["error_among_green"] <= TARGETS["max_error_among_green"]
            and best["coverage_at_risk_0.10"].get("status") == "ok"
            and float(best["coverage_at_risk_0.10"]["coverage"]) >= TARGETS["min_coverage_at_risk_0_10"]
            and float(best["coverage_at_risk_0.20"]["coverage"]) >= TARGETS["min_coverage_at_risk_0_20"]
        ),
    }
    def _jsonable(o: Any) -> Any:
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    meta_path.write_text(json.dumps(meta, indent=2, default=_jsonable), encoding="utf-8")
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"hall_probe_{slug}_{suite}.json"
    out.write_text(json.dumps(meta, indent=2, default=_jsonable), encoding="utf-8")
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / out.name).write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in meta if k not in ("best_selective", "heldout_scores", "heldout_labels")}, indent=2, default=_jsonable))
    print("best_selective", best)
    print(f"Wrote {ckpt} sha={sha[:16]}…")
    return meta


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--suite", choices=("halueval", "truthfulqa"), default="truthfulqa")
    p.add_argument("--max-pairs", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args(argv)
    run(
        model_name=args.model,
        suite=args.suite,
        max_pairs=args.max_pairs,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
