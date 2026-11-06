#!/bin/bash
# On-box watchdog for the overnight GPU fleet.
# Cron every 5m: restart with --resume if the fleet died before rollup.
# Does NOT treat quiet logs as stuck (probe extract can be silent for hours).
set -uo pipefail

ROOT="${ANIMA_ROOT:-$HOME/Anima}"
cd "$ROOT" || exit 1
mkdir -p logs benchmarks/reports

LOG=logs/watchdog.log
FLEET_LOG=logs/overnight_fleet.log
ROLLUP=benchmarks/reports/stage_b_gpu_fleet_rollup.json
STATUS=OVERNIGHT_STATUS.md
LOCK=logs/watchdog.restart.lock
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $*" | tee -a "$LOG"; }

# Done — stop nagging / never relaunch after a finished rollup.
if [ -f "$ROLLUP" ]; then
  log "OK rollup present — watchdog idle"
  # Drop cron self after success (best-effort; ignore failures)
  if crontab -l 2>/dev/null | grep -q watch_overnight_fleet; then
    crontab -l 2>/dev/null | grep -v watch_overnight_fleet | crontab - || true
    log "Removed watchdog crontab entry"
  fi
  exit 0
fi

# Shutdown already armed by fleet launcher
if LC_ALL=C shutdown --show 2>/dev/null | grep -qi power; then
  log "OK shutdown pending — watchdog idle"
  exit 0
fi

export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export ANIMA_LOAD_8BIT=1
export ANIMA_FORCE_CPU=0
export ANIMA_FLEET_SHUTDOWN="${ANIMA_FLEET_SHUTDOWN:-1}"
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"

py_alive=0
sh_alive=0
pgrep -f 'python scripts/run_overnight_gpu_fleet.py' >/dev/null 2>&1 && py_alive=1
pgrep -f 'bash scripts/run_overnight_gpu_fleet.sh' >/dev/null 2>&1 && sh_alive=1

if [ "$py_alive" = "1" ]; then
  # Heartbeat while healthy (for morning diagnostics)
  echo "$(ts) py=1 sh=$sh_alive" > logs/watchdog.heartbeat
  # Partial progress count
  n_ok=$(ls benchmarks/reports/fleet_partial_*.json 2>/dev/null | wc -l | tr -d ' ')
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo '?')
  log "OK fleet running (partials=$n_ok gpu=$gpu)"
  exit 0
fi

# Python gone but wrapper still up — treat as dead
if [ "$sh_alive" = "1" ]; then
  log "WARN wrapper without python — killing stale shell"
  pkill -f 'bash scripts/run_overnight_gpu_fleet.sh' || true
  sleep 2
fi

# Restart gate: avoid thundering herd if cron overlaps
if [ -f "$LOCK" ]; then
  age=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || echo 0) ))
  if [ "$age" -lt 180 ]; then
    log "SKIP restart — lock age ${age}s"
    exit 0
  fi
fi
touch "$LOCK"

log "RESTART fleet (no python, no rollup)"
# shellcheck disable=SC1091
source .venv/bin/activate
nohup env \
  HF_TOKEN="$HF_TOKEN" \
  HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" \
  ANIMA_LOAD_8BIT=1 \
  ANIMA_FORCE_CPU=0 \
  ANIMA_FLEET_SHUTDOWN="$ANIMA_FLEET_SHUTDOWN" \
  HF_HOME="$HF_HOME" \
  bash scripts/run_overnight_gpu_fleet.sh \
  >> logs/overnight_launcher.out 2>&1 &
echo $! > logs/watchdog.last_restart_pid
log "RESTARTED pid=$!"
sleep 8
if pgrep -f 'python scripts/run_overnight_gpu_fleet.py' >/dev/null 2>&1; then
  log "OK python back up"
  rm -f "$LOCK"
  exit 0
fi
log "FAIL python did not start — see $FLEET_LOG / logs/overnight_launcher.out"
rm -f "$LOCK"
exit 1
