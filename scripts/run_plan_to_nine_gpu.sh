#!/usr/bin/env bash
# Plan-to-nine GPU queue: wait for free-gen → TQA n80 encode → hall probe → OP seals
set -euo pipefail
cd ~/Anima
source .venv/bin/activate
export ANIMA_LOAD_8BIT=1
export ANIMA_FORCE_CPU=0
export ANIMA_OUTPUT_ATTENTIONS=0
export HF_HOME="${HF_HOME:-$HOME/Anima/.hf_cache}"
mkdir -p logs benchmarks/reports probes/zoo

echo "[0] Wait for free-gen PID if running"
if [[ -f logs/protocol_g_7b.pid ]]; then
  pid=$(cat logs/protocol_g_7b.pid || true)
  if [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Waiting for free-gen pid=$pid…"
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  fi
fi
# Also wait if process still matches
while pgrep -f "free_gen_protocol.*Qwen2.5-7B" >/dev/null 2>&1; do
  echo "free_gen still running…"; sleep 30
done
echo "Free-gen done or absent. Log tail:"
tail -n 20 logs/protocol_g_7b.log || true
ls -la benchmarks/reports/protocol_g_*stage_b_protocol_g* 2>/dev/null || true

echo "[1] TruthfulQA n=80 claim-encode Qwen2.5-0.5B"
python -m benchmarks.live_guard_eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --benchmark truthfulqa \
  --max-pairs 80 \
  --seed 42 \
  --device cuda \
  --report-suffix tqa_n80_plan9 \
  2>&1 | tee logs/tqa_encode_0_5b.log
# Promote to slug path for OP scripts (additive suffix kept too)
cp -f benchmarks/reports/truthfulqa_live_qwen_qwen2_5-0_5b-instruct_tqa_n80_plan9.json \
      benchmarks/reports/truthfulqa_live_qwen_qwen2_5-0_5b-instruct.json || true

echo "[2] TruthfulQA n=80 claim-encode Qwen2.5-7B (8-bit)"
python -m benchmarks.live_guard_eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --benchmark truthfulqa \
  --max-pairs 80 \
  --seed 42 \
  --device cuda \
  --report-suffix tqa_n80_plan9 \
  2>&1 | tee logs/tqa_encode_7b.log
cp -f benchmarks/reports/truthfulqa_live_qwen_qwen2_5-7b-instruct_tqa_n80_plan9.json \
      benchmarks/reports/truthfulqa_live_qwen_qwen2_5-7b-instruct.json || true

echo "[3] Hallucination hidden-state probe on 0.5B TruthfulQA"
python -m benchmarks.train_hallucination_probe \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --suite truthfulqa \
  --max-pairs 80 \
  --seed 42 \
  --device cuda \
  --epochs 60 \
  2>&1 | tee logs/hall_probe_0_5b_tqa.log

echo "[4] Selective TQA OP + phase2 battery"
python -m benchmarks.protocol_tqa_selective_op \
  --slug qwen_qwen2_5-0_5b-instruct \
  --slug qwen_qwen2_5-7b-instruct \
  2>&1 | tee logs/tqa_selective_op.log
python -m benchmarks.protocol_phase2_9_10_battery 2>&1 | tee logs/phase2_battery.log || true

echo "[5] DONE plan-to-nine GPU queue"
date -u > logs/plan_to_nine_gpu.done
