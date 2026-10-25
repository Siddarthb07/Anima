#!/usr/bin/env python3
"""SE v2 sketch — additive control; does NOT overwrite matched SE v1 or baseline locks.

V2-B (default): answer-span token entropy / mean NLL on the locked claim row's
encoded answer. This varies per row and can rank when v1 SE is question-constant.

V2-A (stub): claim-conditioned multi-sample SE — requires generation; GPU preferred.

Usage (when wiring into live harness):
  python -m benchmarks.semantic_entropy_v2_sketch --help

This module is intentionally a design stub + dry-run helper so the paper can
cite the redesign plan without mutating sealed receipts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "benchmarks" / "reports"


def answer_span_entropy_proxy(logits_or_probs: list[float]) -> float:
    """Shannon entropy of a discrete distribution over vocab slice / tokens."""
    ps = [p for p in logits_or_probs if p > 0]
    if not ps:
        return 0.0
    s = sum(ps)
    ps = [p / s for p in ps]
    return float(-sum(p * math.log(p + 1e-12) for p in ps))


def dry_run_note() -> dict:
    return {
        "schema": "anima.semantic_entropy_v2.sketch",
        "status": "not_run",
        "honesty": {
            "v1_unchanged": True,
            "baseline_lock": "do not overwrite semantic_entropy_matched_*.json or live_guard_*.json",
            "v1_degeneracy": "matched SE AUROC=0.50 when scores are question-constant within pairs",
            "v2b": "answer-span entropy / NLL per claim row (CPU-feasible on 0.5B/1.1B)",
            "v2a": "claim-conditioned K-sample SE (GPU preferred)",
        },
        "see": "docs/paper/FEASIBILITY_7B_AND_SE.md",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-sketch", action="store_true", help="Write sketch JSON under reports/")
    args = ap.parse_args()
    note = dry_run_note()
    print(json.dumps(note, indent=2))
    if args.write_sketch:
        out = REPORTS / "semantic_entropy_v2_sketch.json"
        out.write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
