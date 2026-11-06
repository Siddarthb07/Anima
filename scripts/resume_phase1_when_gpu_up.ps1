# Hands-free: wait until anima-gpu accepts SSH, then push OP script + run resume.
# Usage: powershell -File scripts\resume_phase1_when_gpu_up.ps1
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")

Write-Host "Waiting for anima-gpu SSH..."
$ok = $false
for ($i = 0; $i -lt 120; $i++) {
  ssh -o ConnectTimeout=8 -o BatchMode=yes anima-gpu "echo up" 2>$null
  if ($LASTEXITCODE -eq 0) { $ok = $true; break }
  Start-Sleep -Seconds 30
}
if (-not $ok) { throw "anima-gpu never came up (check AWS start / rate limit)" }

Write-Host "Pushing Phase-1 OP files..."
scp "$root\benchmarks\protocol_op_heldout_phase1.py" anima-gpu:~/Anima/benchmarks/
scp "$root\scripts\resume_phase1_op_gpu.sh" anima-gpu:~/Anima/scripts/
ssh anima-gpu "sed -i 's/\r$//' ~/Anima/scripts/resume_phase1_op_gpu.sh && chmod +x ~/Anima/scripts/resume_phase1_op_gpu.sh && bash ~/Anima/scripts/resume_phase1_op_gpu.sh"

$dst = Join-Path $root "benchmarks\reports"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
scp "anima-gpu:~/Anima/benchmarks/reports/protocol_op_heldout_phase1*.json" $dst
$paper = "C:\Users\siddu\projects\anima-corvex-papers\tnnls\anima\artifact"
if (Test-Path $paper) {
  Copy-Item -Force (Join-Path $dst "protocol_op_heldout_phase1*.json") $paper
  Write-Host "Copied seals to $paper"
}
Write-Host "DONE — stop the EC2 instance in AWS console when finished."
