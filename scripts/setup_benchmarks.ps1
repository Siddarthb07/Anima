# Configure Anima benchmark env vars and verify optional dependencies.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\setup_benchmarks.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=== Anima benchmark setup ===" -ForegroundColor Cyan
Write-Host ""

# 1) Guard fixtures (always available)
$truthful = Join-Path $Root "benchmarks\fixtures\truthfulqa_guard_sample.json"
$halueval = Join-Path $Root "benchmarks\fixtures\halueval_guard_sample.json"
$env:TRUTHFULQA_PATH = $truthful
$env:HALUEVAL_PATH = $halueval
Write-Host "[ok] Guard fixtures:" -ForegroundColor Green
Write-Host "  TRUTHFULQA_PATH=$truthful"
Write-Host "  HALUEVAL_PATH=$halueval"

# 2) Narratives
if (-not $env:NARRATIVES_ROOT) {
    Write-Host ""
    Write-Host "[skip] NARRATIVES_ROOT not set — narratives_holdout / litcoder will skip." -ForegroundColor Yellow
    Write-Host "  Download ds002345: https://openneuro.org/datasets/ds002345"
    Write-Host "  Then: `$env:NARRATIVES_ROOT='C:\path\to\ds002345'"
    Write-Host "  Verify: python scripts\download_narratives.py --root `$env:NARRATIVES_ROOT"
} else {
    python scripts\download_narratives.py --root $env:NARRATIVES_ROOT
}

# 3) Brain-Score (optional; often needs Python 3.11+)
if ($env:SKIP_BRAINSCORE -ne "1") {
    $bs = python -c "import brainscore_language" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[hint] brainscore-language not installed. Try: pip install brainscore-language" -ForegroundColor Yellow
        Write-Host "  Or set `$env:SKIP_BRAINSCORE='1' to skip in manifests."
        $env:SKIP_BRAINSCORE = "1"
    } else {
        Write-Host "[ok] brainscore_language importable" -ForegroundColor Green
    }
}

# 4) TRIBEv2 runtime
if ($env:ANIMA_TRIBE_MODE -ne "runtime") {
    $env:ANIMA_TRIBE_MODE = "surrogate"
}
$tribe = python -c "import tribev2" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[hint] tribev2 not installed — tribe_reference uses surrogate only." -ForegroundColor Yellow
} else {
    Write-Host "[ok] tribev2 importable (set ANIMA_TRIBE_MODE=runtime for full compare)" -ForegroundColor Green
}

# 5) Probe zoo reminder
Write-Host ""
Write-Host "=== Models / Ollama ===" -ForegroundColor Cyan
Write-Host "Anima uses Hugging Face ids only (see core/layer_config.py)."
Write-Host "Ollama is NOT supported. Llama/Mistral/Qwen in layer_config have NO shipped probe .pt — train with anima train-text / anima train."
Write-Host "Docs: docs/MODELS_AND_ZOO.md"
Write-Host ""

# 6) Persist example env file if missing
$envFile = Join-Path $Root ".env.benchmark"
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $Root ".env.benchmark.example") $envFile
    Write-Host "Created .env.benchmark from example (edit NARRATIVES_ROOT there)." -ForegroundColor Green
}

Write-Host ""
Write-Host "Run benchmarks:" -ForegroundColor Cyan
Write-Host '  python -m benchmarks.run_all --model hf-internal-testing/tiny-random-gpt2 --tiers internal,external,external_text,external_guard'
