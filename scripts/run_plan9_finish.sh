#!/usr/bin/env bash
set -euo pipefail
cd ~/Anima
source .venv/bin/activate
export ANIMA_LOAD_8BIT=1 ANIMA_FORCE_CPU=0 ANIMA_OUTPUT_ATTENTIONS=0
export HF_HOME="${HF_HOME:-$HOME/Anima/.hf_cache}"
mkdir -p logs

cp -f benchmarks/reports/truthfulqa_live_qwen_qwen2_5-0_5b-instruct_tqa_n80_plan9.json \
      benchmarks/reports/truthfulqa_live_qwen_qwen2_5-0_5b-instruct.json || true
cp -f benchmarks/reports/truthfulqa_live_qwen_qwen2_5-7b-instruct_tqa_n80_plan9.json \
      benchmarks/reports/truthfulqa_live_qwen_qwen2_5-7b-instruct.json || true

echo "[1] Retrain TQA hall probe (suite-specific ckpt)"
python -m benchmarks.train_hallucination_probe \
  --model Qwen/Qwen2.5-0.5B-Instruct --suite truthfulqa \
  --max-pairs 80 --seed 42 --device cuda --epochs 80 \
  2>&1 | tee logs/hall_probe_tqa_final.log

echo "[2] Retrain HaluEval hall probe (secondary; length-contaminated expected)"
python -m benchmarks.train_hallucination_probe \
  --model Qwen/Qwen2.5-0.5B-Instruct --suite halueval \
  --max-pairs 80 --seed 42 --device cuda --epochs 60 \
  2>&1 | tee logs/hall_probe_halu_final.log || true

echo "[3] Ablation + OP + channel + battery"
python -m benchmarks.protocol_hall_ablation 2>&1 | tee logs/hall_ablation.log
python -m benchmarks.protocol_tqa_selective_op \
  --slug qwen_qwen2_5-0_5b-instruct \
  --slug qwen_qwen2_5-7b-instruct \
  2>&1 | tee logs/tqa_selective_op_final.log
python -m benchmarks.protocol_second_channel_probe 2>&1 | tee logs/second_channel.log || true
python -m benchmarks.protocol_phase2_9_10_battery 2>&1 | tee logs/phase2_battery_final.log || true

date -u > logs/plan9_finish.done
echo DONE
ls -la probes/zoo/*hall_probe* benchmarks/reports/hall_probe_* benchmarks/reports/protocol_hall_ablation* benchmarks/reports/protocol_tqa_selective* 2>/dev/null
