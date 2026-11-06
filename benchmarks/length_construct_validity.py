"""Length construct-validity ownership table (CPU, sealed rows).

Owns the council finding that length→label AUROC can dominate HaluEval.
Additive sealed receipt; does not overwrite locked live JSON.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.guard_metrics import auroc_from_scores, bootstrap_auroc_ci
from benchmarks.length_balanced_and_baselines import (
    greedy_length_match,
    inv_gap,
    residualized_composite,
)
from benchmarks.pairing_bootstrap_audit import score_of
from benchmarks.protocol_m_multi_recipe_contract import LOCK_MATRIX, REPORTS, _rows


def _pack(rows: list[dict[str, Any]], label_key: str) -> dict[str, Any]:
    labels = [1 if bool(r[label_key]) else 0 for r in rows]
    lens = [float(len(str(r.get("answer") or ""))) for r in rows]
    comp = [score_of(r, "composite") for r in rows]
    probe = [score_of(r, "probe") for r in rows]
    fused = [score_of(r, "fused") for r in rows]
    inv = [inv_gap(r) for r in rows]
    resid = residualized_composite(rows)
    return {
        "n": len(rows),
        "n_pos": int(sum(labels)),
        "answer_length_auroc": auroc_from_scores(lens, labels),
        "answer_length_ci95": bootstrap_auroc_ci(lens, labels, B=2000, seed=42),
        "composite_auroc": auroc_from_scores(comp, labels),
        "composite_residualized_auroc": auroc_from_scores(resid, labels),
        "probe_auroc": auroc_from_scores(probe, labels),
        "fused_auroc": auroc_from_scores(fused, labels),
        "inv_logit_gap_auroc": auroc_from_scores(inv, labels),
        "length_dominates_composite": auroc_from_scores(lens, labels)
        >= auroc_from_scores(comp, labels) + 0.2,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=REPORTS / "length_construct_validity.json",
    )
    args = ap.parse_args()

    models: dict[str, Any] = {}
    flags: list[dict[str, Any]] = []

    for slug, suite, label_key, live_name, _fix in LOCK_MATRIX:
        live_path = REPORTS / live_name
        if not live_path.exists():
            models.setdefault(slug, {})[suite] = {"status": "missing"}
            continue
        rows = _rows(live_path)
        full = _pack(rows, label_key)
        matched = greedy_length_match(rows, label_key)
        match_pack = _pack(matched, label_key) if matched else {"status": "empty"}
        block = {
            "status": "ok",
            "source_live": live_name,
            "full_locked": full,
            "greedy_length_matched": match_pack,
            "construct_threat": full["answer_length_auroc"] >= 0.8,
        }
        models.setdefault(slug, {})[suite] = block
        if block["construct_threat"]:
            flags.append(
                {
                    "model": slug,
                    "suite": suite,
                    "length_auroc": full["answer_length_auroc"],
                    "composite_auroc": full["composite_auroc"],
                    "residualized": full["composite_residualized_auroc"],
                }
            )

    out = {
        "schema": "system_a.length_construct_validity.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "honesty": {
            "no_new_forward_passes": True,
            "locked_n80_untouched": True,
            "note": (
                "High length→label AUROC is a construct threat on that suite; "
                "residualization and length-matched tables are disclosures, not fixes."
            ),
        },
        "rollup": {
            "n_construct_threat_cells": len(flags),
            "threat_cells": flags,
        },
        "models": models,
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"construct-threat cells: {len(flags)}")


if __name__ == "__main__":
    main()
