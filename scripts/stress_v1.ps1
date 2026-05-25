# Anima v1 gate: fast pytest + optional API health (no Playwright).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\stress_v1.ps1
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

$env:CUDA_VISIBLE_DEVICES = ""
$env:ANIMA_FORCE_CPU = "1"
$env:RUN_HF_TESTS = "0"

Write-Host "=== pytest (no distilgpt2 HF download) ===" -ForegroundColor Cyan
python -m pytest -q -k "not distilgpt2" --tb=short
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== manifest + split smoke ===" -ForegroundColor Cyan
python -m pytest tests/test_manifest.py tests/test_models_api.py -q --tb=short
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($env:RUN_API_SMOKE -eq "1") {
    Write-Host "=== API health (expects uvicorn on 8010) ===" -ForegroundColor Cyan
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8010/health" -TimeoutSec 5
        if ($r.status -ne "ok") { throw "health not ok" }
        $models = Invoke-RestMethod -Uri "http://127.0.0.1:8010/models" -TimeoutSec 30
        Write-Host "models:" $models.models.Count
    } catch {
        Write-Host "API smoke skipped or failed: $_" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "=== stress_v1 OK ===" -ForegroundColor Green
