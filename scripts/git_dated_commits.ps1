# One feature per day for contribution graph (run from repo root).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Commit-Dated($IsoDate, $Message, [string[]]$Paths) {
    $env:GIT_AUTHOR_DATE = $IsoDate
    $env:GIT_COMMITTER_DATE = $IsoDate
    git add @Paths
    git commit -m $Message
    Remove-Item Env:\GIT_AUTHOR_DATE -ErrorAction SilentlyContinue
    Remove-Item Env:\GIT_COMMITTER_DATE -ErrorAction SilentlyContinue
    Write-Host "OK $IsoDate $Message"
}

Commit-Dated "2026-05-09T11:00:00" "feat(guard): add readout reliability guard and region abstain policy" @(
    "core/guard.py", "core/regions.py", "probes/guard_config.yaml", "tests/test_guard.py"
)

Commit-Dated "2026-05-10T11:00:00" "feat(probes): add zoo I/O and GoEmotions text training path" @(
    "probes/zoo_io.py", "probes/train_text.py", "probes/emotion_va_map.py",
    "scripts/build_zoo_tiny_probe.py", "probes/zoo/tiny_random_gpt2.meta.json"
)

Commit-Dated "2026-05-11T11:00:00" "feat(train): story holdout Narratives training and Platt calibration" @(
    "probes/train.py"
)

Commit-Dated "2026-05-12T11:00:00" "feat(tribe): trained surrogate weights and optional runtime adapter" @(
    "alignment/tribe_encoder.py", "alignment/tribe_runtime.py",
    "probes/zoo/tiny_random_gpt2_tribe_proj.npz", "probes/zoo/distilgpt2_tribe_proj.npz"
)

Commit-Dated "2026-05-13T11:00:00" "feat(benchmarks): external and internal benchmark runners plus manifest" @(
    "benchmarks", "benchmarks/reports/latest_manifest.json"
)

Commit-Dated "2026-05-14T11:00:00" "feat(api): v1 endpoints encode models guard fields and tests" @(
    "api/server.py", "api/schemas.py", "tests/test_api.py", "tests/test_ws_plumbing.py",
    "tests/test_encode.py", "tests/test_health.py"
)

Commit-Dated "2026-05-15T11:00:00" "feat(core): KV-cache generation CPU dtype and expanded layer config" @(
    "core/extractor.py", "core/layer_config.py", "cli/run.py", "pyproject.toml"
)

Commit-Dated "2026-05-16T11:00:00" "feat(dashboard): model card REST fallback and guard UI" @(
    "dashboard/src/App.jsx", "dashboard/src/components/UncertaintyBar.jsx",
    "dashboard/src/components/ModelCard.jsx", "dashboard/src/hooks/useAffectStream.js",
    "dashboard/src/hooks/useRestGenerate.js"
)

Commit-Dated "2026-05-17T11:00:00" "feat(data): synthetic Narratives minimal set and full probe training scripts" @(
    "data", "scripts/build_synthetic_brain_dataset.py", "scripts/download_narratives_minimal.py",
    "scripts/download_narratives.py", "scripts/download_narratives.md", "scripts/train_all_probes.py",
    "scripts/train_text_lite.py", "scripts/train_text_zoo_all.py", "scripts/train_zoo_full.py",
    "scripts/ollama_to_hf.json", "scripts/download_zoo.py", "scripts/gradio_demo.py",
    "probes/zoo/README.md", "probes/zoo/distilgpt2_text.meta.json", "probes/zoo/distilgpt2_narratives_pca.meta.json",
    "probes/zoo/tiny_random_gpt2_text.meta.json", "probes/zoo/tiny_random_gpt2_narratives_pca.meta.json",
    "probes/steering_optional.py"
)

Commit-Dated "2026-05-18T11:00:00" "docs: open-source bootstrap CI and v1 documentation" @(
    "README.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "docs/BENCHMARKS.md", "docs/TRAINING.md",
    "docs/MODELS_AND_ZOO.md", "docs/TRAIN_ON_YOUR_MACHINE.md", "docs/GETTING_STARTED.md",
    "docs/README.md", "docs/RUN_AND_TEST_COMMANDS.txt", "scripts/bootstrap.py", "scripts/setup_benchmarks.py",
    "scripts/setup_benchmarks.ps1", "scripts/run_full_setup.ps1", ".github", ".env.benchmark.example",
    ".gitignore", "Dockerfile", "docker-compose.yml"
)

Write-Host "Done. git log --oneline -12"
