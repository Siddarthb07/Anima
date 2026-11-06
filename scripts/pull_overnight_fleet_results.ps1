# Run tomorrow morning after the instance is started (if it auto-stopped).
# Usage: powershell -File scripts\pull_overnight_fleet_results.ps1

$dst = Join-Path $PSScriptRoot "..\benchmarks\reports"
New-Item -ItemType Directory -Force -Path $dst | Out-Null

Write-Host "Checking anima-gpu..."
ssh -o ConnectTimeout=20 anima-gpu "test -f ~/Anima/OVERNIGHT_STATUS.md && cat ~/Anima/OVERNIGHT_STATUS.md || echo STILL_RUNNING_OR_MISSING; tail -n 30 ~/Anima/logs/overnight_fleet.log"

Write-Host "Pulling reports..."
scp "anima-gpu:~/Anima/benchmarks/reports/stage_b_gpu_fleet_rollup.json" $dst
scp "anima-gpu:~/Anima/benchmarks/reports/stage_b_7b_claim_encode_receipt.json" $dst
scp "anima-gpu:~/Anima/benchmarks/reports/*stage_b_fleet*" $dst
scp "anima-gpu:~/Anima/OVERNIGHT_STATUS.md" (Join-Path $PSScriptRoot "..\OVERNIGHT_STATUS.md")

Write-Host "Done. Next: fold into tnnls/anima paper package."
