#!/bin/bash
# Continuous supervisor (60s) — faster than cron for overnight recovery.
# Idempotent: only one copy via flock.
set -uo pipefail
ROOT="${ANIMA_ROOT:-$HOME/Anima}"
cd "$ROOT" || exit 1
mkdir -p logs
LOCKFILE=logs/deep_supervisor.flock
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "supervisor already running"
  exit 0
fi
echo $$ > logs/deep_supervisor.pid
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) supervisor start pid=$$" >> logs/deep_supervisor.log
while true; do
  /bin/bash "$ROOT/scripts/watch_deep_waiver.sh" >> logs/deep_supervisor.log 2>&1 || true
  # If rollup exists, slow down but keep cancelling shutdown
  if [ -f "$ROOT/benchmarks/reports/deep_waiver/deep_waiver_rollup.json" ]; then
    sleep 300
  else
    sleep 60
  fi
done
