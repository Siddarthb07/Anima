# Benchmarks (Anima v1)

Anima reports numbers on **external** suites where possible, plus **internal** smoke checks. Results land in `benchmarks/reports/<run>/manifest.json`, `benchmarks/reports/latest_manifest.json`, and `benchmarks/reports/latest_<slug>_manifest.json` (e.g. `latest_distilgpt2_manifest.json`). Manifests include `manifest_schema_version: 1`.

## One-time setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_benchmarks.ps1
# or: python scripts/setup_benchmarks.py
```

This sets guard fixture paths, checks optional packages, and documents `NARRATIVES_ROOT`. See [MODELS_AND_ZOO.md](MODELS_AND_ZOO.md) for **Ollama vs Hugging Face** and which models have real probe weights.

## Run

```powershell
# Internal only (fast, no Narratives)
anima benchmark --model hf-internal-testing/tiny-random-gpt2 --tiers internal

# Full external (needs data / optional packages)
$env:NARRATIVES_ROOT="C:\path\to\ds002345"
$env:SKIP_BRAINSCORE="1"   # unless brainscore-language installed (Py3.11+)
python -m benchmarks.run_all --model distilgpt2 --tiers internal,external,external_text,external_guard
```

## Tier A — Brain alignment (external)

| Benchmark | Metric | Notes |
|-----------|--------|-------|
| **Narratives holdout** | Val Pearson r (valence/arousal) vs held-out stories | `benchmarks/run_narratives_encoding.py`; split in `benchmarks/splits/narratives_holdout.json` |
| **LITcoder-style ridge** | Word-rate baseline r on holdout | Same runner + baseline fields |
| **Brain-Score Language** | Normalized score 0–1 | `pip install brainscore-language`; may need Python 3.11 env |

## Tier B — Text / guard (external)

| Benchmark | Metric | Notes |
|-----------|--------|-------|
| **GoEmotions** | Pearson r vs mapped VA | Requires `*_text.pt` checkpoint |
| **HaluEval guard** | Abstain AUROC on fixture | Default: `benchmarks/fixtures/halueval_guard_sample.json` |
| **TruthfulQA guard** | Abstain accuracy / AUROC | Default: `benchmarks/fixtures/truthfulqa_guard_sample.json` |
| **TRIBE reference** | Surrogate vs runtime ROI correlation | `ANIMA_TRIBE_MODE`; optional `tribev2` |

## Tier C — Internal

| Benchmark | Purpose |
|-----------|---------|
| `run_smoke.py` | Extract + probe load without full training |

## Reproducing README tables

```bash
anima benchmark --model hf-internal-testing/tiny-random-gpt2 --tiers internal,external,external_text,external_guard
# distilgpt2 needs ~8GB+ RAM to load the LM; if OOM, use meta export:
python -m benchmarks.export_meta_manifest --model distilgpt2
```

- Tiny: `benchmarks/reports/latest_manifest.json`
- DistilGPT-2: `benchmarks/reports/latest_distilgpt2_manifest.json`

Do not cherry-pick: failed/skipped entries include a `reason` field.

## TRIBE reference

Compare surrogate vs optional runtime via `ANIMA_TRIBE_MODE=blend` and `alignment/tribe_runtime.py`. Full TRIBE weights are optional (`pip install anima[tribe]` when Meta package is available).
