#!/usr/bin/env bash
# Re-train hall probe (save heldout_scores), ablate SysA/G'/hall/channel, reseal OP.
set -euo pipefail
cd ~/Anima
source .venv/bin/activate
export ANIMA_LOAD_8BIT=1
export ANIMA_FORCE_CPU=0
export ANIMA_OUTPUT_ATTENTIONS=0
export HF_HOME="${HF_HOME:-$HOME/Anima/.hf_cache}"
mkdir -p logs benchmarks/reports probes/zoo

# Ensure n80 live JSON promoted
if [[ -f benchmarks/reports/truthfulqa_live_qwen_qwen2_5-0_5b-instruct_tqa_n80_plan9.json ]]; then
  cp -f benchmarks/reports/truthfulqa_live_qwen_qwen2_5-0_5b-instruct_tqa_n80_plan9.json \
        benchmarks/reports/truthfulqa_live_qwen_qwen2_5-0_5b-instruct.json
fi
if [[ -f benchmarks/reports/truthfulqa_live_qwen_qwen2_5-7b-instruct_tqa_n80_plan9.json ]]; then
  cp -f benchmarks/reports/truthfulqa_live_qwen_qwen2_5-7b-instruct_tqa_n80_plan9.json \
        benchmarks/reports/truthfulqa_live_qwen_qwen2_5-7b-instruct.json
fi

echo "[1] Retrain hall probe 0.5B TQA n=80 (heldout_scores)"
python -m benchmarks.train_hallucination_probe \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --suite truthfulqa \
  --max-pairs 80 \
  --seed 42 \
  --device cuda \
  --epochs 80 \
  2>&1 | tee logs/hall_probe_retrain.log

echo "[2] Optional: hall probe on HaluEval n=80 (secondary)"
python -m benchmarks.train_hallucination_probe \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --suite halueval \
  --max-pairs 80 \
  --seed 42 \
  --device cuda \
  --epochs 60 \
  2>&1 | tee logs/hall_probe_halu.log || true

echo "[3] Ablation + selective OP + channel + battery"
python -m benchmarks.protocol_hall_ablation 2>&1 | tee logs/hall_ablation.log
python -m benchmarks.protocol_tqa_selective_op \
  --slug qwen_qwen2_5-0_5b-instruct \
  --slug qwen_qwen2_5-7b-instruct \
  2>&1 | tee logs/tqa_selective_op2.log
python -m benchmarks.protocol_second_channel_probe 2>&1 | tee logs/second_channel.log || true
python -m benchmarks.protocol_phase2_9_10_battery 2>&1 | tee logs/phase2_battery2.log || true

echo "[4] DONE ablation"
date -u > logs/plan9_ablation.done
ls -la benchmarks/reports/hall_probe_* protocol*tqa* protocol_hall* 2>/dev/null || true
ls -la benchmarks/reports/protocol_hall_ablation_rollup.json benchmarks/reports/protocol_tqa_selective_op_rollup.json
