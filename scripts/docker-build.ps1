# Build Docker images only (do not start containers).
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root
Write-Host "Building anima API + dashboard + model-pull images..." -ForegroundColor Cyan
# Docker Compose writes progress to stderr; do not treat that as a terminating error on Windows.
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
docker compose --profile stack --profile pull build
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $prevEap
if ($exitCode -ne 0) { exit $exitCode }
Write-Host "Build complete. Start later with: .\scripts\docker-up.ps1 qwen" -ForegroundColor Green
