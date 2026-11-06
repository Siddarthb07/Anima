#!/usr/bin/env bash
# Confirmatory B–C fleet on T4: length-bal Halu n=300 + hall probe TQA n=300
# across every model that fits with a text probe on disk.
# Usage: bash ~/Anima/scripts/run_multimodel_bc_fleet.sh
set -euo pipefail
cd "$HOME/Anima"
source .venv/bin/activate
export ANIMA_FORCE_CPU=0 ANIMA_OUTPUT_ATTENTIONS=0 ANIMA_PREFER_TEXT_PROBE=1
export HF_HOME="${HF_HOME:-$HOME/Anima/.hf_cache}"
mkdir -p logs benchmarks/reports probes/zoo artifact/prereg

ROLLUP="benchmarks/reports/multimodel_bc_fleet_rollup.json"
LOG="logs/multimodel_bc_fleet.log"
: >"$LOG"

# T4-safe confirmatory set (7B fp16 = Phase D on g5 — listed but skipped here)
MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
)

python - <<'PY' 2>&1 | tee -a "$LOG"
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.guard_metrics import score_abstain_accuracy
from benchmarks.length_balanced_pairs import build_halueval_balanced
from benchmarks.live_guard_eval import evaluate_pairs
from benchmarks.train_hallucination_probe import run as train_hall
from probes.zoo_io import resolve_checkpoint_slug

ROOT = Path.home() / "Anima"
REPORTS = ROOT / "benchmarks" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

device = "cuda"
pairs = build_halueval_balanced(max_pairs=300, max_ratio=1.25, seed=42)
print(f"[fleet] built length-bal pairs n={len(pairs)}", flush=True)

rollup: dict = {
    "schema": "anima.multimodel_bc_fleet.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "protocol": "L_length_balanced_v1 + hall_probe_truthfulqa_n300",
    "n_pairs_lenbal": len(pairs),
    "seed": 42,
    "device": device,
    "shipping_claim": False,
    "cells": {},
}

for model in MODELS:
    cell: dict = {"model": model, "status": "pending"}
    print(f"\n======== FLEET {model} ========", flush=True)
    ckpt, slug, meta = resolve_checkpoint_slug(model)
    cell["probe_slug"] = slug
    cell["probe_origin"] = meta.get("probe_origin")
    cell["probe_checkpoint"] = str(ckpt) if ckpt else None
    if ckpt is None or meta.get("probe_origin") == "random":
        cell["status"] = "skipped_no_text_probe"
        print(f"[skip] no text probe for {model}", flush=True)
        rollup["cells"][model] = cell
        continue
    try:
        # C1 length-balanced Halu encode
        rows = evaluate_pairs(model, pairs, label_key="is_hallucination", device=device)
        n_nan = sum(
            1
            for r in rows
            if r.get("composite_score") != r.get("composite_score")
        )
        metrics = score_abstain_accuracy(
            rows, label_key="is_hallucination", with_ci=True, ci_B=2000, ci_seed=42
        )
        safe = (
            model.lower()
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
        )
        out = REPORTS / f"halueval_live_{safe}_lenbal_n300.json"
        payload = {
            "rows": rows,
            "metrics": metrics,
            "protocol": "L_length_balanced_v1",
            "seed": 42,
            "model_name": model,
            "probe_meta": {
                "probe_origin": meta.get("probe_origin"),
                "checkpoint": str(ckpt),
            },
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        cell["lenbal"] = {
            "path": str(out),
            "n_samples": metrics.get("n_samples"),
            "auroc_composite": metrics.get("auroc_composite"),
            "auroc_ci95": metrics.get("auroc_ci95"),
            "never_abstains": metrics.get("never_abstains"),
            "n_nan_composite": n_nan,
        }
        print(
            f"[C1] {model} auroc={metrics.get('auroc_composite')} n_nan={n_nan}",
            flush=True,
        )

        # B1 hall probe TQA
        hall = train_hall(
            model_name=model,
            suite="truthfulqa",
            max_pairs=300,
            seed=42,
            device=device,
            epochs=80,
        )
        cell["hall_tqa"] = {
            "heldout_auroc": hall.get("heldout_auroc"),
            "length_to_label_auroc": hall.get("length_to_label_auroc"),
            "hits_9_gates": hall.get("hits_9_gates"),
            "sha256": hall.get("sha256"),
            "n_test": hall.get("n_test"),
        }
        cell["status"] = "ok"
    except Exception as e:
        cell["status"] = "error"
        cell["error"] = repr(e)
        cell["traceback"] = traceback.format_exc()[-2000:]
        print(f"[ERROR] {model}: {e!r}", flush=True)
        traceback.print_exc()
    rollup["cells"][model] = cell
    (REPORTS / "multimodel_bc_fleet_rollup.json").write_text(
        json.dumps(rollup, indent=2, default=str), encoding="utf-8"
    )

ok = sum(1 for c in rollup["cells"].values() if c.get("status") == "ok")
print(f"\n[fleet] done ok={ok}/{len(MODELS)}", flush=True)
print(json.dumps({k: v.get("status") for k, v in rollup["cells"].items()}, indent=2))
PY

date -u > logs/multimodel_bc_fleet.done
echo FLEET_DONE
