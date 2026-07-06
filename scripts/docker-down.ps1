# Stop all Anima Docker stacks and free RAM.
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root
docker compose --profile stack --profile pull down
Write-Host "All Anima containers stopped." -ForegroundColor Green
