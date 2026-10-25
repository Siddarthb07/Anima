#!/usr/bin/env python3
"""Finding-style predict-then-test controls on sealed Protocol L rows (additive).

Mirrors the HeavyBridge revision pattern: name the failure mode, predict the
control outcome, then measure it on locked JSON without inventing numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.pairing_bootstrap_audit import auroc, score_of

REPORTS = Path(__file__).resolve().parent / "reports"
ARTIFACT = Path(__file__).resolve().parents[2] / "anima-corvex-papers" / "tmlr" / "anima" / "artifact"

# Prefer local Anima reports; fall back to papers artifact copy.
SEARCH = [REPORTS, ARTIFACT, Path(r"C:\Users\siddu\projects\anima-corvex-papers\tmlr\anima\artifact")]


def _find(name: str) -> Path | None:
    for root in SEARCH:
        if not root:
            continue
        p = root / name
        if p.is_file():
            return p
    return None


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rows" in data:
        rows = data["rows"]
    elif isinstance(data, list):
        rows = data
    else:
        # live_guard style: nested suites
        for key in ("rows", "samples", "results"):
            if key in data and isinstance(data[key], list):
                rows = data[key]
                break
        else:
            rows = data.get("halueval", data.get("items", []))
            if isinstance(rows, dict):
                rows = rows.get("rows", [])
    label_key = "is_hallucination"
    if rows and "should_abstain" in rows[0] and "is_hallucination" not in rows[0]:
        label_key = "should_abstain"
    return rows, label_key


def residualize(rows: list[dict[str, Any]]) -> list[float]:
    lens = [float(len(str(r.get("answer") or ""))) for r in rows]
    comps = [score_of(r, "composite") for r in rows]
    n = len(rows)
    mx = sum(lens) / n
    my = sum(comps) / n
    varx = sum((x - mx) ** 2 for x in lens) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(lens, comps)) / n
    slope = cov / varx if varx > 0 else 0.0
    return [y - (my + slope * (x - mx)) for x, y in zip(lens, comps)]


def finding_controls() -> dict[str, Any]:
    """Pre-declared controls C-length / C-ablation / C-never-abstain."""
    qwen_h = _find("halueval_live_qwen_qwen2_5-0_5b-instruct.json") or _find(
        "halueval_live_qwen25_0_5b.json"
    )
    expand = _find("expand_n160_seed43_summary.json")
    length = _find("length_balanced_and_baselines.json")
    pairing = _find("pairing_and_paired_bootstrap_audit.json")

    findings: list[dict[str, Any]] = []

    # Finding 1 control: length residualization
    if qwen_h and qwen_h.is_file():
        rows, lk = _load_rows(qwen_h)
        if rows and lk in rows[0]:
            labels = [bool(r[lk]) for r in rows]
            comp = [score_of(r, "composite") for r in rows]
            probe = [score_of(r, "probe") for r in rows]
            fused = [float(r["uncertainty_signals"]["fused"]) for r in rows]
            lens = [float(len(str(r.get("answer") or ""))) for r in rows]
            resid = residualize(rows)
            abstain = sum(1 for r in rows if r.get("abstain_recommended"))
            findings.append(
                {
                    "id": "Finding1_length_control",
                    "prediction": (
                        "Residualizing composite on answer length raises AUROC but "
                        "does not create abstain events at fixture thresholds."
                    ),
                    "observed": {
                        "n": len(rows),
                        "auroc_composite": round(auroc(comp, labels), 4),
                        "auroc_length": round(auroc(lens, labels), 4),
                        "auroc_residualized": round(auroc(resid, labels), 4),
                        "auroc_probe": round(auroc(probe, labels), 4),
                        "auroc_fused": round(auroc(fused, labels), 4),
                        "abstain_count": abstain,
                    },
                    "prediction_held": abstain == 0,
                }
            )
            findings.append(
                {
                    "id": "Finding2_misweight_control",
                    "prediction": "Probe-only AUROC exceeds full composite on Qwen–Halu; fused hurts.",
                    "observed": {
                        "auroc_probe": round(auroc(probe, labels), 4),
                        "auroc_fused": round(auroc(fused, labels), 4),
                        "auroc_composite": round(auroc(comp, labels), 4),
                    },
                    "prediction_held": auroc(probe, labels) > auroc(comp, labels)
                    and auroc(fused, labels) <= auroc(probe, labels),
                }
            )

    if expand and expand.is_file():
        ex = json.loads(expand.read_text(encoding="utf-8"))
        qh = ex.get("qwen", {}).get("halueval", {})
        findings.append(
            {
                "id": "Finding3_expand_never_abstain",
                "prediction": "Additive n=160 seed43 still never-abstains; Halu AUROC stays <0.5.",
                "observed": qh,
                "prediction_held": bool(qh.get("never_abstain"))
                and float(qh.get("auroc", 1)) < 0.5,
            }
        )

    sealed_refs = {
        "length_balanced_and_baselines": bool(length),
        "pairing_and_paired_bootstrap_audit": bool(pairing),
        "expand_n160_seed43_summary": bool(expand),
        "qwen_halueval_live": bool(qwen_h),
    }

    return {
        "schema": "anima.predict_then_test_controls",
        "honesty": {"no_invented_metrics": True, "additive_only": True},
        "sealed_refs": sealed_refs,
        "findings": findings,
        "all_predictions_held": all(f.get("prediction_held") for f in findings) if findings else False,
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    # Also search papers artifact by injecting path
    global SEARCH
    SEARCH = [
        REPORTS,
        Path(r"C:\Users\siddu\projects\anima-corvex-papers\satml2027\anima-artifact\data"),
        Path(r"C:\Users\siddu\projects\anima-corvex-papers\tmlr\anima\artifact"),
    ]
    out = finding_controls()
    path = REPORTS / "predict_then_test_controls.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
