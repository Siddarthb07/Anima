#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Anima"
mkdir -p logs artifact/prereg benchmarks/reports probes/zoo
# kill prior run if any
if pgrep -f run_phase_bcd_gpu.sh >/dev/null 2>&1; then
  echo "already running:"
  pgrep -af run_phase_bcd_gpu.sh || true
  exit 0
fi
nohup bash scripts/run_phase_bcd_gpu.sh > logs/phase_bcd_gpu_stdout.log 2>&1 &
echo "PID=$!"
sleep 3
tail -n 50 logs/phase_bcd_gpu_stdout.log || true
