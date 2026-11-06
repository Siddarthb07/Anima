#!/usr/bin/env bash
# Overnight multi-guard claim-encode. Cached HF models only. Leaves the box up.
set -euo pipefail
cd /home/ubuntu/Anima
mkdir -p logs benchmarks/reports
export HF_HOME=/home/ubuntu/Anima/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export ANIMA_FORCE_CPU=0
export ANIMA_OUTPUT_ATTENTIONS=0
export ANIMA_LOAD_8BIT=0
export ANIMA_PREFER_TEXT_PROBE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -f logs/multi_guard_claim_encode.pid ]]; then
  old=$(cat logs/multi_guard_claim_encode.pid || true)
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "already running pid=${old}"
    exit 0
  fi
fi

nohup .venv/bin/python -u benchmarks/multi_guard_claim_encode.py \
  --device cuda \
  --seeds 42 43 44 \
  --max-pairs 160 \
  --benchmark both \
  --out benchmarks/reports/multi_guard_claim_encode_rollup.json \
  > logs/multi_guard_claim_encode.log 2>&1 &
echo $! > logs/multi_guard_claim_encode.pid
echo "started pid=$(cat logs/multi_guard_claim_encode.pid)"
