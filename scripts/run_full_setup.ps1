# Full local setup: benchmark env + CPU-tier text probes + internal benchmarks.
# Requires: pip install -e ".[dev]" and enough RAM / page file for distilgpt2 (~2GB free).

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:CUDA_VISIBLE_DEVICES = ""
$env:ANIMA_FORCE_CPU = "1"
$env:SKIP_BRAINSCORE = "1"

Write-Host "1) Benchmark setup" -ForegroundColor Cyan
python scripts/setup_benchmarks.py

Write-Host "`n2a) Tiny model text probe (always fits RAM)" -ForegroundColor Cyan
python scripts/train_text_lite.py --max-samples 60 --epochs 8

Write-Host "`n2b) Full CPU tier (distilgpt2 + open proxies — needs ~4GB+ free RAM)" -ForegroundColor Cyan
python scripts/train_text_zoo_all.py --tier cpu --max-samples 100 --epochs 6

Write-Host "`n3) Internal + guard benchmarks" -ForegroundColor Cyan
python -m benchmarks.run_all --model hf-internal-testing/tiny-random-gpt2 --tiers internal,external_text,external_guard

Write-Host "`nDone. See probes/zoo/train_text_report.json" -ForegroundColor Green
Write-Host "For Llama-3-8B / Mistral-7B: GPU machine + huggingface-cli login + ANIMA_TRAIN_LARGE=1"
