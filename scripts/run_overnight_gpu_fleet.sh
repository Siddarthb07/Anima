#!/bin/bash
# Hands-free overnight launcher. Auto-stops the instance when finished
# so g4dn credits do not burn idle overnight.
set -uo pipefail
cd ~/Anima
source .venv/bin/activate
# Prefer env; else VidhiSetu/local HF cache token on this host.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export ANIMA_LOAD_8BIT=1
export ANIMA_FORCE_CPU=0
export HF_HOME="${HF_HOME:-$HOME/Anima/.hf_cache}"
mkdir -p logs benchmarks/reports

LOG=logs/overnight_fleet.log
echo "=== overnight fleet start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

# Default to --resume so a restart continues after partial progress.
python scripts/run_overnight_gpu_fleet.py --resume "$@" >>"$LOG" 2>&1
rc=$?
echo "=== overnight fleet exit rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

# Mark done for morning pull
echo "$rc" > logs/overnight_fleet.exit
cp -f OVERNIGHT_STATUS.md logs/OVERNIGHT_STATUS.md 2>/dev/null || true

# Only auto-stop after a real fleet finish (rollup exists). Set
# ANIMA_FLEET_SHUTDOWN=0 to keep the box running.
if [ "${ANIMA_FLEET_SHUTDOWN:-1}" = "1" ] && [ -f benchmarks/reports/stage_b_gpu_fleet_rollup.json ]; then
  echo "Scheduling poweroff in 3 minutes…" | tee -a "$LOG"
  sudo shutdown -h +3 "Anima overnight fleet finished (rc=$rc)"
else
  echo "Skipping auto-shutdown (rc=$rc, no rollup or ANIMA_FLEET_SHUTDOWN=0)" | tee -a "$LOG"
fi

exit $rc
