#!/bin/bash
# Install/refresh the overnight fleet watchdog crontab (every 5 minutes).
set -euo pipefail
ROOT="${ANIMA_ROOT:-$HOME/Anima}"
cd "$ROOT"
chmod +x scripts/watch_overnight_fleet.sh scripts/run_overnight_gpu_fleet.sh
# Normalize CRLF if copied from Windows
sed -i 's/\r$//' scripts/watch_overnight_fleet.sh scripts/run_overnight_gpu_fleet.sh scripts/install_fleet_watchdog.sh

LINE="*/5 * * * * ANIMA_ROOT=$ROOT /bin/bash $ROOT/scripts/watch_overnight_fleet.sh >>$ROOT/logs/watchdog.cron.out 2>&1"
tmp=$(mktemp)
(crontab -l 2>/dev/null | grep -v watch_overnight_fleet || true) >"$tmp"
echo "$LINE" >>"$tmp"
crontab "$tmp"
rm -f "$tmp"
echo "Installed crontab:"
crontab -l | grep watch_overnight_fleet
# Immediate tick
/bin/bash "$ROOT/scripts/watch_overnight_fleet.sh"
echo "First watchdog tick done — see logs/watchdog.log"
