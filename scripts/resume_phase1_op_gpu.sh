#!/usr/bin/env bash
# Resume Phase-1 OP GPU work when anima-gpu is up (AWS rate-limit cleared).
# Usage (on EC2 after scp of latest Anima tree):
#   bash scripts/resume_phase1_op_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export ANIMA_LOAD_8BIT=1
export ANIMA_FORCE_CPU=0
export HF_HOME="${HF_HOME:-$HOME/Anima/.hf_cache}"
mkdir -p logs benchmarks/reports/deep_waiver

echo "[1/4] Ensure deep-waiver live rows exist for Qwen2.5-7B (n=160 preferred)"
if [[ ! -f benchmarks/reports/halueval_live_qwen_qwen2_5-7b-instruct_deep_waiver_v1.json ]]; then
  echo "Missing deep_waiver_v1 live JSON — re-running deep waiver encode for 7B only"
  python scripts/run_overnight_deep_waiver.py --models Qwen/Qwen2.5-7B-Instruct --resume \
    > logs/phase1_op_deep_waiver.log 2>&1 || true
fi

echo "[2/4] Symlink / copy deep-waiver live → slug path for Protocol OP"
DW=benchmarks/reports/halueval_live_qwen_qwen2_5-7b-instruct_deep_waiver_v1.json
DST=benchmarks/reports/halueval_live_qwen_qwen2_5-7b-instruct.json
if [[ -f "$DW" ]]; then
  cp -f "$DW" "$DST"
  echo "Copied $DW → $DST"
else
  echo "WARN: still no deep-waiver live JSON; OP will use whatever $DST exists"
fi

echo "[3/4] Seal Protocol OP (CPU on live rows; no new forwards if rows present)"
python -m benchmarks.protocol_op_heldout_phase1 \
  --slug qwen_qwen2_5-0_5b-instruct \
  --slug qwen_qwen2_5-7b-instruct \
  | tee logs/phase1_op_seal.log

echo "[4/4] Done. Pull to laptop:"
echo "  scp anima-gpu:~/Anima/benchmarks/reports/protocol_op_heldout_phase1*.json <paper>/artifact/"
echo "Then STOP the instance."
