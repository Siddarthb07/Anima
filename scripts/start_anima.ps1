# Requires: pip install -e . ; dashboard npm install ; PowerShell execution policy allows scripts.
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$dash = Join-Path $root "dashboard"

Write-Host "Repo: $root"
Write-Host "Opening API: uvicorn on http://127.0.0.1:8010 (match dashboard/.env VITE_API_HTTP_TARGET)"
Start-Process powershell -WorkingDirectory $root -ArgumentList @(
    "-NoExit",
    "-Command",
    "python -m uvicorn api.server:app --host 127.0.0.1 --port 8010"
)

Start-Sleep -Seconds 2

Write-Host "Opening dashboard: npm run dev on http://127.0.0.1:5173"
Start-Process powershell -WorkingDirectory $dash -ArgumentList @(
    "-NoExit",
    "-Command",
    "npm run dev -- --host 127.0.0.1 --port 5173"
)

Write-Host ""
Write-Host "When both show ready: open http://127.0.0.1:5173"
Write-Host "Default model is hf-internal-testing/tiny-random-gpt2 (reliable on modest RAM)."
