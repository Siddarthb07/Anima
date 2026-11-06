#!/usr/bin/env bash
# Watchdog for run_multimodel_bc_fleet.sh — writes STATUS every poll; exits on done/dead.
# Usage: nohup bash ~/Anima/scripts/watch_multimodel_fleet.sh >> ~/Anima/logs/multimodel_watchdog.log 2>&1 &
set -euo pipefail
ROOT="$HOME/Anima"
LOG="$ROOT/logs/multimodel_bc_fleet.log"
DONE="$ROOT/logs/multimodel_bc_fleet.done"
ROLLUP="$ROOT/benchmarks/reports/multimodel_bc_fleet_rollup.json"
STATUS="$ROOT/logs/multimodel_fleet_STATUS.txt"
POLL="${POLL_SEC:-60}"

status_once() {
  local ts alive roll
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if pgrep -f run_multimodel_bc_fleet.sh >/dev/null 2>&1; then
    alive=RUNNING
  else
    alive=DEAD
  fi
  roll="no_rollup"
  if [[ -f "$ROLLUP" ]]; then
    roll="$(python3 -c "
import json
from pathlib import Path
d=json.loads(Path('$ROLLUP').read_text())
parts=[]
for k,v in d.get('cells',{}).items():
    short=k.split('/')[-1][:28]
    lb=(v.get('lenbal') or {}).get('auroc_composite')
    h=(v.get('hall_tqa') or {}).get('heldout_auroc')
    parts.append(f\"{short}:{v.get('status')} lenbal={lb} hall={h}\")
print(' | '.join(parts) if parts else 'empty')
" 2>/dev/null || echo parse_err)"
  fi
  local tailc
  tailc="$(tail -n 3 "$LOG" 2>/dev/null | tr '\n' ' ' | cut -c1-200 || true)"
  {
    echo "ts=$ts alive=$alive"
    echo "rollup=$roll"
    echo "tail=$tailc"
    if [[ -f "$DONE" ]]; then echo "DONE_FILE=1"; else echo "DONE_FILE=0"; fi
  } | tee "$STATUS"
}

while true; do
  status_once
  if [[ -f "$DONE" ]]; then
    echo "WATCHDOG_FLEET_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 0
  fi
  if ! pgrep -f run_multimodel_bc_fleet.sh >/dev/null 2>&1; then
    # give rollup a moment to flush
    sleep 5
    status_once
    if [[ -f "$DONE" ]]; then
      echo "WATCHDOG_FLEET_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      exit 0
    fi
    echo "WATCHDOG_FLEET_DEAD $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 1
  fi
  sleep "$POLL"
done
