"""
Protocol R (optimistic same-set) and Protocol R' (held-out) threshold retune.

Offline on locked live guard rows — no new LM runs.
Does not overwrite probes/guard_config.yaml.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from benchmarks.guard_metrics import auroc_from_scores

REPORTS = Path(__file__).resolve().parent.parent / "benchmarks" / "reports"


def _load_arrays(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["rows"]
    y: list[int] = []
    questions: list[str] = []
    for r in rows:
        if r.get("is_hallucination") is not None:
            y.append(1 if r.get("is_hallucination") else 0)
        elif r.get("should_abstain") is not None:
            y.append(1 if r.get("should_abstain") else 0)
        else:
            y.append(0)
        questions.append(str(r.get("question") or ""))
    comp, fused, probe, hedge = [], [], [], []
    for r in rows:
        us = r.get("uncertainty_signals") or {}
        aff = r.get("affect") or {}
        fused.append(float(us.get("fused", us.get("fused_uncertainty", r.get("composite_score", 0)))))
        probe.append(
            float(
                us.get(
                    "probe",
                    us.get("probe_uncertainty", aff.get("uncertainty", 0.5)),
                )
            )
        )
        reasons = str(r.get("reasons", "")).lower()
        hedge.append(
            1
            if (us.get("lexical_hedge") or us.get("hedging") or "hedge" in reasons)
            else 0
        )
        comp.append(float(r["composite_score"]))
    return {
        "y": np.asarray(y, dtype=int),
        "comp": np.asarray(comp, float),
        "fused": np.asarray(fused, float),
        "probe": np.asarray(probe, float),
        "hedge": np.asarray(hedge, int),
        "questions": questions,
        "n": len(rows),
        "source_path": str(path),
    }


def _decide(
    fused: np.ndarray,
    probe: np.ndarray,
    hedge: np.ndarray,
    y: np.ndarray,
    tf: float,
    tp: float,
) -> tuple[int, float, float, float]:
    high_f = fused >= tf
    high_p = probe >= tp
    votes = high_f.astype(int) + high_p.astype(int) + hedge
    abstain = (fused >= tf) | (votes >= 2)
    acc = float(((abstain == 1) & (y == 1) | (abstain == 0) & (y == 0)).mean()) if len(y) else 0.0
    recall = float(abstain[y == 1].mean()) if y.sum() else 0.0
    prec = float(y[abstain == 1].mean()) if abstain.sum() else 0.0
    return int(abstain.sum()), acc, recall, prec


def sweep_thresholds(
    fused: np.ndarray,
    probe: np.ndarray,
    hedge: np.ndarray,
    y: np.ndarray,
    *,
    grid: int = 71,
) -> dict[str, Any]:
    a0, acc0, rec0, prec0 = _decide(fused, probe, hedge, y, 0.82, 0.78)
    best = None
    for tf in np.linspace(0.2, 0.9, grid):
        for tp in np.linspace(0.2, 0.9, grid):
            a, acc, recall, prec = _decide(fused, probe, hedge, y, float(tf), float(tp))
            key = (acc, recall, -a)
            if best is None or key > best[0]:
                best = (key, float(tf), float(tp), a, acc, recall, prec)
    assert best is not None
    return {
        "default_tf": 0.82,
        "default_tp": 0.78,
        "default_abstain": a0,
        "default_acc": round(acc0, 4),
        "default_recall": round(rec0, 4),
        "default_prec": round(prec0, 4),
        "best_tf": round(best[1], 3),
        "best_tp": round(best[2], 3),
        "best_abstain": best[3],
        "best_acc": round(best[4], 4),
        "best_recall": round(best[5], 4),
        "best_prec": round(best[6], 4),
    }


def question_holdout_split(
    questions: list[str],
    *,
    seed: int = 42,
    frac_tune: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for tune/test with no question leakage."""
    uniq = sorted(set(questions))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_tune = max(1, int(round(len(uniq) * frac_tune)))
    tune_q = {uniq[i] for i in order[:n_tune]}
    tune_mask = np.array([q in tune_q for q in questions], dtype=bool)
    test_mask = ~tune_mask
    if not test_mask.any():
        # tiny fallback
        test_mask[-1] = True
        tune_mask[-1] = False
    return tune_mask, test_mask


def evaluate_suite(
    path: Path,
    *,
    seed: int = 42,
    grid: int = 71,
    held_out: bool = True,
) -> dict[str, Any]:
    D = _load_arrays(path)
    y, fused, probe, hedge, comp = D["y"], D["fused"], D["probe"], D["hedge"], D["comp"]
    base = {
        "n": int(D["n"]),
        "source_path": D["source_path"],
        "composite_auroc": auroc_from_scores(comp.tolist(), y.tolist()),
        "probe_auroc": auroc_from_scores(probe.tolist(), y.tolist()),
        "fused_auroc": auroc_from_scores(fused.tolist(), y.tolist()),
        "fused_min": round(float(fused.min()), 4),
        "fused_max": round(float(fused.max()), 4),
        "probe_min": round(float(probe.min()), 4),
        "probe_max": round(float(probe.max()), 4),
    }
    # Protocol R — optimistic same-set
    r = sweep_thresholds(fused, probe, hedge, y, grid=grid)
    r["protocol"] = "R"
    r["optimistic"] = True
    r["held_out"] = False
    r["caveat"] = "thresholds tuned on same live rows (optimistic; not held-out)"
    out: dict[str, Any] = {**base, "protocol_r": r}

    if held_out:
        tune_m, test_m = question_holdout_split(D["questions"], seed=seed)
        # ensure both classes present on tune when possible
        r_tune = sweep_thresholds(fused[tune_m], probe[tune_m], hedge[tune_m], y[tune_m], grid=grid)
        tf, tp = r_tune["best_tf"], r_tune["best_tp"]
        a, acc, recall, prec = _decide(fused[test_m], probe[test_m], hedge[test_m], y[test_m], tf, tp)
        # also default thresholds on test fold
        a_d, acc_d, rec_d, prec_d = _decide(
            fused[test_m], probe[test_m], hedge[test_m], y[test_m], 0.82, 0.78
        )
        out["protocol_r_prime"] = {
            "protocol": "R_prime",
            "optimistic": False,
            "held_out": True,
            "seed": seed,
            "n_tune": int(tune_m.sum()),
            "n_test": int(test_m.sum()),
            "n_tune_questions": len({D["questions"][i] for i in range(len(D["questions"])) if tune_m[i]}),
            "n_test_questions": len({D["questions"][i] for i in range(len(D["questions"])) if test_m[i]}),
            "tuned_on": "question_grouped_fold_A",
            "reported_on": "question_grouped_fold_B",
            "best_tf": tf,
            "best_tp": tp,
            "test_abstain": a,
            "test_acc": round(acc, 4),
            "test_recall": round(recall, 4),
            "test_prec": round(prec, 4),
            "default_test_abstain": a_d,
            "default_test_acc": round(acc_d, 4),
            "default_test_recall": round(rec_d, 4),
            "default_test_prec": round(prec_d, 4),
            "tune_fold_best_acc": r_tune["best_acc"],
            "caveat": "thresholds frozen on held-out question fold; primary operating-point claim",
        }
    return out


def run(
    *,
    slug: str,
    reports_dir: Path = REPORTS,
    seed: int = 42,
    held_out: bool = True,
) -> dict[str, Any]:
    suites = {
        "halueval": reports_dir / f"halueval_live_{slug}.json",
        "truthfulqa": reports_dir / f"truthfulqa_live_{slug}.json",
    }
    report: dict[str, Any] = {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_slug": slug,
        "seed": seed,
        "ieee_note": "R is optimistic secondary; R' is primary held-out operating point",
        "suites": {},
    }
    for name, path in suites.items():
        if not path.is_file():
            report["suites"][name] = {"status": "missing", "path": str(path)}
            continue
        block = evaluate_suite(path, seed=seed, held_out=held_out)
        block["status"] = "ok"
        report["suites"][name] = block
    return report


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Protocol R / R' threshold retune (offline)")
    p.add_argument("--slug", required=True, help="e.g. qwen_qwen2_5-0_5b-instruct")
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-held-out", action="store_true")
    args = p.parse_args(argv)
    report = run(
        slug=args.slug,
        reports_dir=args.reports_dir,
        seed=args.seed,
        held_out=not args.no_held_out,
    )
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    out_r = args.reports_dir / f"protocol_r_{args.slug}.json"
    out_rp = args.reports_dir / f"protocol_r_prime_{args.slug}.json"
    out_r.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # slim R' file for primary claim tables
    slim = {
        "status": report["status"],
        "generated_at": report["generated_at"],
        "model_slug": args.slug,
        "seed": args.seed,
        "protocol": "R_prime",
        "suites": {
            k: {
                "status": v.get("status"),
                "n": v.get("n"),
                "composite_auroc": v.get("composite_auroc"),
                **(v.get("protocol_r_prime") or {"status": "skipped"}),
            }
            for k, v in report["suites"].items()
        },
    }
    out_rp.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"Wrote {out_r}")
    print(f"Wrote {out_rp}")
    for name, block in report["suites"].items():
        if block.get("status") != "ok":
            print(f"  {name}: {block.get('status')}")
            continue
        r = block["protocol_r"]
        rp = block.get("protocol_r_prime") or {}
        print(
            f"  {name}: R best_acc={r['best_acc']} "
            f"R' test_acc={rp.get('test_acc')} "
            f"(n_test={rp.get('n_test')})"
        )


if __name__ == "__main__":
    main()
