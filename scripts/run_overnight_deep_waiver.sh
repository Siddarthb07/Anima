#!/bin/bash
# Hands-free deep waiver overnight. NEVER auto-shutdown.
set -uo pipefail
cd ~/Anima
source .venv/bin/activate

# Prefer HF token from cache
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export ANIMA_LOAD_8BIT=1
export ANIMA_FORCE_CPU=0
export ANIMA_FLEET_SHUTDOWN=0
export ANIMA_OUTPUT_ATTENTIONS="${ANIMA_OUTPUT_ATTENTIONS:-0}"
export HF_HOME="${HF_HOME:-$HOME/Anima/.hf_cache}"
mkdir -p logs benchmarks/reports/deep_waiver

# Cancel any pending poweroff
sudo shutdown -c 2>/dev/null || true

LOG=logs/deep_waiver.log
echo "=== deep waiver start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
echo "ANIMA_OUTPUT_ATTENTIONS=$ANIMA_OUTPUT_ATTENTIONS ANIMA_FLEET_SHUTDOWN=$ANIMA_FLEET_SHUTDOWN" | tee -a "$LOG"

python scripts/run_overnight_deep_waiver.py --resume "$@" >>"$LOG" 2>&1
rc=$?
echo "=== deep waiver exit rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
echo "$rc" > logs/deep_waiver.exit
cp -f DEEP_WAIVER_STATUS.md logs/DEEP_WAIVER_STATUS.md 2>/dev/null || true

# Explicitly do NOT shut down.
echo "Deep suite finished (rc=$rc). Instance left running for canvas/pull." | tee -a "$LOG"
exit $rc
