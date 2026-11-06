"""Deep channel / waiver analysis over live claim-encode row JSON.

Does not invent metrics. NaN channels are reported with nan_rate and excluded
from AUROC (finite-only), with explicit counts.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from benchmarks.guard_metrics import auroc_from_scores, bootstrap_auroc_ci


def _finite(xs: Sequence[Optional[float]]) -> list[float]:
    out: list[float] = []
    for x in xs:
        try:
            v = float(x)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if v == v and v not in (float("inf"), float("-inf")):
            out.append(v)
    return out


def _nan_rate(xs: Sequence[Any]) -> float:
    n = len(xs)
    if n == 0:
        return 1.0
    bad = 0
    for x in xs:
        try:
            v = float(x)
            if v != v or v in (float("inf"), float("-inf")):
                bad += 1
        except (TypeError, ValueError):
            bad += 1
    return round(bad / n, 4)


def _channel_auroc(
    scores: Sequence[Any], labels: Sequence[int]
) -> dict[str, Any]:
    pairs = []
    for s, y in zip(scores, labels):
        try:
            v = float(s)
        except (TypeError, ValueError):
            continue
        if v == v and v not in (float("inf"), float("-inf")):
            pairs.append((v, int(y)))
    if len(pairs) < 8 or len({y for _, y in pairs}) < 2:
        return {
            "n_finite": len(pairs),
            "auroc": None,
            "auroc_flip": None,
            "ci95": None,
            "polarity": "insufficient_finite",
        }
    vals = [p[0] for p in pairs]
    labs = [p[1] for p in pairs]
    a = auroc_from_scores(vals, labs)
    af = auroc_from_scores([-v for v in vals], labs)
    best = a if a >= af else af
    polarity = (
        "aligned"
        if a >= 0.55
        else ("anti_aligned" if af >= 0.55 else "near_chance")
    )
    return {
        "n_finite": len(pairs),
        "auroc": a,
        "auroc_flip": af,
        "best_auroc": best,
        "ci95": bootstrap_auroc_ci(vals, labs, B=1000, seed=42),
        "ci95_flip": bootstrap_auroc_ci([-v for v in vals], labs, B=1000, seed=42),
        "polarity": polarity,
        "mean": round(sum(vals) / len(vals), 4),
        "stdev": round(
            math.sqrt(sum((v - sum(vals) / len(vals)) ** 2 for v in vals) / len(vals)), 4
        ),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def _length_residual_auroc(
    scores: Sequence[Any], labels: Sequence[int], lengths: Sequence[int]
) -> dict[str, Any]:
    """AUROC of score residualized on answer length (OLS)."""
    rows = []
    for s, y, L in zip(scores, labels, lengths):
        try:
            v = float(s)
        except (TypeError, ValueError):
            continue
        if v != v:
            continue
        rows.append((v, int(y), float(L)))
    if len(rows) < 16 or len({r[1] for r in rows}) < 2:
        return {"auroc_residual": None, "n": len(rows)}
    # simple OLS: score ~ a + b*length
    n = len(rows)
    mean_L = sum(r[2] for r in rows) / n
    mean_s = sum(r[0] for r in rows) / n
    var_L = sum((r[2] - mean_L) ** 2 for r in rows)
    if var_L < 1e-9:
        resid = [r[0] for r in rows]
    else:
        cov = sum((r[2] - mean_L) * (r[0] - mean_s) for r in rows)
        b = cov / var_L
        a = mean_s - b * mean_L
        resid = [r[0] - (a + b * r[2]) for r in rows]
    labs = [r[1] for r in rows]
    length_auroc = auroc_from_scores([r[2] for r in rows], labs)
    return {
        "n": n,
        "length_auroc": length_auroc,
        "auroc_residual": auroc_from_scores(resid, labs),
        "auroc_residual_flip": auroc_from_scores([-x for x in resid], labs),
        "raw_auroc": auroc_from_scores([r[0] for r in rows], labs),
    }


def _waiver_bins(
    scores: Sequence[Any], labels: Sequence[int], *, n_bins: int = 5
) -> list[dict[str, Any]]:
    """Where ranking wavers: hallucination rate by score quintile."""
    pairs = []
    for s, y in zip(scores, labels):
        try:
            v = float(s)
        except (TypeError, ValueError):
            continue
        if v == v:
            pairs.append((v, int(y)))
    if len(pairs) < n_bins * 4:
        return []
    pairs.sort(key=lambda t: t[0])
    bin_size = len(pairs) // n_bins
    out = []
    for i in range(n_bins):
        chunk = pairs[i * bin_size : (i + 1) * bin_size if i < n_bins - 1 else len(pairs)]
        if not chunk:
            continue
        rate = sum(y for _, y in chunk) / len(chunk)
        out.append(
            {
                "bin": i,
                "n": len(chunk),
                "score_lo": round(chunk[0][0], 4),
                "score_hi": round(chunk[-1][0], 4),
                "positive_rate": round(rate, 4),
            }
        )
    return out


def analyze_rows(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
) -> dict[str, Any]:
    labels = [1 if bool(r.get(label_key)) else 0 for r in rows]
    abstain = [bool(r.get("abstain_recommended")) for r in rows]
    lengths = [len(str(r.get("answer") or "")) for r in rows]

    def col(path: str) -> list[Any]:
        out = []
        for r in rows:
            if path == "composite_score":
                out.append(r.get("composite_score"))
            elif path.startswith("affect."):
                out.append((r.get("affect") or {}).get(path.split(".", 1)[1]))
            else:
                out.append((r.get("uncertainty_signals") or {}).get(path))
        return out

    entropy = col("entropy")
    gap = col("logit_gap")
    attn = col("attn_entropy")
    fused = col("fused")
    probe_u = col("affect.uncertainty")
    composite = col("composite_score")
    inv_gap = []
    for g in gap:
        try:
            v = float(g)
            inv_gap.append(1.0 / (1.0 + v) if v == v else float("nan"))
        except (TypeError, ValueError):
            inv_gap.append(float("nan"))

    channels = {
        "entropy": _channel_auroc(entropy, labels),
        "logit_gap": _channel_auroc(gap, labels),
        "inv_logit_gap": _channel_auroc(inv_gap, labels),
        "attn_entropy": _channel_auroc(attn, labels),
        "fused": _channel_auroc(fused, labels),
        "probe_uncertainty": _channel_auroc(probe_u, labels),
        "composite": _channel_auroc(composite, labels),
    }

    # Prefer best polarity channel for waiver bins
    primary = "composite"
    best_score = -1.0
    for name, ch in channels.items():
        b = ch.get("best_auroc")
        if b is not None and b > best_score and ch.get("n_finite", 0) >= 16:
            best_score = b
            primary = name
    primary_scores = {
        "entropy": entropy,
        "logit_gap": gap,
        "inv_logit_gap": inv_gap,
        "attn_entropy": attn,
        "fused": fused,
        "probe_uncertainty": probe_u,
        "composite": composite,
    }[primary]
    # Use flipped scores for bins if flip is the better polarity
    ch_primary = channels[primary]
    use_flip = (ch_primary.get("auroc_flip") or 0) > (ch_primary.get("auroc") or 0)
    bin_scores = (
        [(-float(s) if s == s else float("nan")) if _ok(s) else float("nan") for s in primary_scores]
        if use_flip
        else primary_scores
    )

    return {
        "n_rows": len(rows),
        "label_balance": dict(Counter(labels)),
        "never_abstains": not any(abstain),
        "abstain_rate": round(sum(1 for a in abstain if a) / max(1, len(abstain)), 4),
        "nan_rates": {
            "entropy": _nan_rate(entropy),
            "logit_gap": _nan_rate(gap),
            "attn_entropy": _nan_rate(attn),
            "fused": _nan_rate(fused),
            "probe_uncertainty": _nan_rate(probe_u),
            "composite": _nan_rate(composite),
        },
        "channels": channels,
        "length_confound": {
            "composite": _length_residual_auroc(composite, labels, lengths),
            "fused": _length_residual_auroc(fused, labels, lengths),
            "probe_uncertainty": _length_residual_auroc(probe_u, labels, lengths),
            "inv_logit_gap": _length_residual_auroc(inv_gap, labels, lengths),
        },
        "waiver_bins": {
            "primary_channel": primary,
            "used_flip": use_flip,
            "bins": _waiver_bins(bin_scores, labels, n_bins=5),
        },
        "headline": {
            "best_channel": primary,
            "best_auroc": ch_primary.get("best_auroc"),
            "polarity": ch_primary.get("polarity"),
            "reported_composite_auroc": channels["composite"].get("auroc"),
            "reported_composite_flip": channels["composite"].get("auroc_flip"),
        },
    }


def _ok(s: Any) -> bool:
    try:
        v = float(s)
        return v == v
    except (TypeError, ValueError):
        return False


def analyze_row_file(path: Path, *, label_key: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    out = analyze_rows(rows, label_key=label_key)
    out["source_file"] = str(path)
    out["label_key"] = label_key
    return out


def write_analysis(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
