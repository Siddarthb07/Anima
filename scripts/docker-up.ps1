# Start Anima in Docker with one model profile (qwen | distil | tiny).
# Usage: .\scripts\docker-up.ps1 qwen
param(
    [Parameter(Position = 0)]
    [ValidateSet("qwen", "distil", "tiny")]
    [string]$Profile = "qwen"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

$defaultModel = switch ($Profile) {
    "qwen" { "Qwen/Qwen2.5-0.5B-Instruct" }
    "distil" { "distilgpt2" }
    "tiny" { "TinyLlama/TinyLlama-1.1B-Chat-v1.0" }
}

$env:VITE_DEFAULT_MODEL = $defaultModel
$env:ANIMA_WARMUP_MODEL = $defaultModel
$env:VITE_WS_BASE = "ws://localhost:8080"
$env:VITE_API_HTTP_TARGET = "http://localhost:8080"
Write-Host "  (First run: python scripts/download_zoo.py  then  docker compose --profile pull run --rm model-pull)" -ForegroundColor DarkGray
Write-Host "Starting stack '$Profile' -> $defaultModel" -ForegroundColor Cyan
Write-Host "  dashboard http://localhost:8080  |  API http://localhost:8010" -ForegroundColor DarkGray
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
docker compose --profile stack up -d --build
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $prevEap
if ($exitCode -ne 0) { exit $exitCode }
Write-Host "Done. Stop with: .\scripts\docker-down.ps1" -ForegroundColor Green
