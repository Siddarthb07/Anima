#!/usr/bin/env bash
# Phase B/C/D GPU queue — run AFTER instance is up and this repo is synced.
# Usage: bash ~/Anima/scripts/run_phase_bcd_gpu.sh
set -euo pipefail
cd ~/Anima
source .venv/bin/activate
export ANIMA_FORCE_CPU=0 ANIMA_OUTPUT_ATTENTIONS=0
export HF_HOME="${HF_HOME:-$HOME/Anima/.hf_cache}"
mkdir -p logs benchmarks/reports probes/zoo artifact/prereg

echo "[B1] Hall probe train/val/test on TQA n=300 (needs HF + GPU)"
python -m benchmarks.train_hallucination_probe \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --suite truthfulqa --max-pairs 300 --seed 42 --device cuda --epochs 80 \
  2>&1 | tee logs/phase_b_hall_tqa_n300.log

echo "[C1] Length-balanced HaluEval encode n=300"
python - <<'PY'
from benchmarks.length_balanced_pairs import build_halueval_balanced
from benchmarks.live_guard_eval import evaluate_pairs
from benchmarks.guard_metrics import score_abstain_accuracy
import json
from pathlib import Path
pairs = build_halueval_balanced(max_pairs=300, max_ratio=1.25, seed=42)
print("built", len(pairs))
rows = evaluate_pairs("Qwen/Qwen2.5-0.5B-Instruct", pairs, label_key="is_hallucination", device="cuda")
metrics = score_abstain_accuracy(rows, label_key="is_hallucination", with_ci=True, ci_B=2000, ci_seed=42)
out = Path("benchmarks/reports/halueval_live_qwen_qwen2_5-0_5b-instruct_lenbal_n300.json")
out.write_text(json.dumps({"rows": rows, "metrics": metrics, "protocol": "L_length_balanced_v1", "seed": 42}, indent=2))
print("wrote", out, "n", metrics.get("n_samples"), "auroc", metrics.get("auroc_composite"))
PY

echo "[B2] Transfer gap on new cells"
python -m benchmarks.protocol_transfer_gap 2>&1 | tee logs/phase_a_transfer_gap.log || true

echo "[D note] 7B fp16 parity is a SEPARATE job — needs larger GPU; see run_phase_d_7b_fp16.sh"
date -u > logs/phase_bcd_gpu.done
echo DONE
