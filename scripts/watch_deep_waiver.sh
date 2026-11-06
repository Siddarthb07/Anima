#!/bin/bash
# Hardened watchdog for deep waiver overnight.
# Goals: keep instance alive, keep suite running, surface progress for exceptional readings.
# NEVER schedules shutdown.
set -uo pipefail
ROOT="${ANIMA_ROOT:-$HOME/Anima}"
cd "$ROOT" || exit 1
mkdir -p logs benchmarks/reports/deep_waiver

LOG=logs/deep_watchdog.log
ROLLUP=benchmarks/reports/deep_waiver/deep_waiver_rollup.json
STATUS=DEEP_WAIVER_STATUS.md
LOCK=logs/deep_watchdog.restart.lock
HB=logs/deep_watchdog.heartbeat
STALL_SECS="${DEEP_WATCHDOG_STALL_SECS:-10800}"  # 3h silent log before force-restart

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $*" | tee -a "$LOG"; }

# 1) Never let a leftover poweroff fire
sudo shutdown -c 2>/dev/null || true

# 2) Done — stay idle, leave machine up for pull/canvas
if [ -f "$ROLLUP" ]; then
  log "OK deep rollup present — idle (NO shutdown)"
  n_ok=$(python3 - <<'PY'
import json
from pathlib import Path
p=Path('benchmarks/reports/deep_waiver/deep_waiver_rollup.json')
d=json.loads(p.read_text())
print(d.get('n_ok'), d.get('n_failed'), d.get('finished_at','?'))
PY
  2>/dev/null || echo "?")
  echo "$(ts) DONE rollup ok_failed_finished=$n_ok" > "$HB"
  # Keep cron so we keep cancelling shutdown overnight; do not remove.
  exit 0
fi

export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export ANIMA_LOAD_8BIT="${ANIMA_LOAD_8BIT:-1}"
export ANIMA_FORCE_CPU=0
export ANIMA_FLEET_SHUTDOWN=0
export ANIMA_OUTPUT_ATTENTIONS="${ANIMA_OUTPUT_ATTENTIONS:-0}"
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"

py_alive=0
sh_alive=0
pgrep -f 'python scripts/run_overnight_deep_waiver.py' >/dev/null 2>&1 && py_alive=1
pgrep -f 'bash scripts/run_overnight_deep_waiver.sh' >/dev/null 2>&1 && sh_alive=1

partials=$(ls benchmarks/reports/deep_waiver/partial_*.json 2>/dev/null | wc -l | tr -d ' ')
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo '?')

# Progress snippet from latest partial / log
progress=$(python3 - <<'PY'
from pathlib import Path
import json, re
parts=sorted(Path('benchmarks/reports/deep_waiver').glob('partial_*.json'))
bits=[]
for p in parts[-3:]:
    try:
        d=json.loads(p.read_text())
        h=((d.get('analysis') or {}).get('suites') or {}).get('halueval') or {}
        head=h.get('headline') or {}
        bits.append(f"{d.get('model','?').split('/')[-1]}:{d.get('status')}:best={head.get('best_auroc')}")
    except Exception:
        bits.append(p.name)
log=Path('logs/deep_waiver.log')
phase='?'
if log.exists():
    t=log.read_text(errors='replace')
    for line in reversed(t.splitlines()):
        s=line.strip('\r')
        if 'phase' in s.lower() or s.startswith('#####') or 'HaluEval' in s or 'health' in s:
            phase=s[:120]
            break
print((' | '.join(bits)) if bits else 'no_partials', '||', phase)
PY
2>/dev/null || echo 'progress=?')

if [ "$py_alive" = "1" ]; then
  # Stall detection: log mtime very old while process allegedly alive
  stall=0
  if [ -f logs/deep_waiver.log ]; then
    age=$(( $(date +%s) - $(stat -c %Y logs/deep_waiver.log) ))
    if [ "$age" -gt "$STALL_SECS" ]; then
      stall=1
    fi
  fi
  echo "$(ts) py=1 sh=$sh_alive partials=$partials stall=$stall gpu=$gpu :: $progress" > "$HB"
  if [ "$stall" = "1" ]; then
    log "WARN stall log_age>${STALL_SECS}s — killing for resume restart"
    pkill -f 'python scripts/run_overnight_deep_waiver.py' || true
    pkill -f 'bash scripts/run_overnight_deep_waiver.sh' || true
    sleep 3
    # fall through to restart
  else
    log "OK deep running (partials=$partials gpu=$gpu) $progress"
    exit 0
  fi
fi

# Wrapper without python = zombie
if [ "$sh_alive" = "1" ] && [ "$py_alive" = "0" ]; then
  log "WARN wrapper without python — killing stale shell"
  pkill -f 'bash scripts/run_overnight_deep_waiver.sh' || true
  sleep 2
fi

if [ -f "$LOCK" ]; then
  age=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || echo 0) ))
  if [ "$age" -lt 120 ]; then
    log "SKIP restart — lock age ${age}s"
    exit 0
  fi
fi
touch "$LOCK"

log "RESTART deep waiver (no python, no rollup) partials=$partials"
# shellcheck disable=SC1091
source .venv/bin/activate
nohup env \
  HF_TOKEN="$HF_TOKEN" \
  HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" \
  ANIMA_LOAD_8BIT="$ANIMA_LOAD_8BIT" \
  ANIMA_FORCE_CPU=0 \
  ANIMA_FLEET_SHUTDOWN=0 \
  ANIMA_OUTPUT_ATTENTIONS="$ANIMA_OUTPUT_ATTENTIONS" \
  HF_HOME="$HF_HOME" \
  bash scripts/run_overnight_deep_waiver.sh --resume \
  >> logs/deep_launcher.out 2>&1 &
echo $! > logs/deep_watchdog.last_restart_pid
log "RESTARTED pid=$!"
sleep 12
if pgrep -f 'python scripts/run_overnight_deep_waiver.py' >/dev/null 2>&1; then
  log "OK deep python back up"
  rm -f "$LOCK"
  exit 0
fi
log "FAIL deep python did not start — see logs/deep_launcher.out / logs/deep_waiver.log"
rm -f "$LOCK"
exit 1
