# Training probes (Anima v1)

## Text-emotion probe (GoEmotions)

Fast path — no fMRI download:

```powershell
pip install -e ".[dev]"
anima train-text --model distilgpt2 --max-samples 1500
```

Writes `probes/zoo/distilgpt2_text.pt` and `distilgpt2_text.meta.json`.

## Text + brain on tiny model (one command)

```powershell
python scripts/download_narratives_minimal.py
python scripts/train_all_probes.py
```

Uses `data/narratives_minimal/` (synthetic BOLD, Narratives layout) when ds002345 is not installed. Outputs `tiny_random_gpt2_text.pt` and `tiny_random_gpt2_narratives_pca.pt`. Holdout split: [`benchmarks/splits/narratives_holdout.json`](../benchmarks/splits/narratives_holdout.json) (train `pieman`/`tunnel`, holdout `lucy`).

## Narratives fMRI probe (real OpenNeuro data)

1. Download [OpenNeuro ds002345](https://openneuro.org/datasets/ds002345) (see `scripts/download_narratives.md`) or `python scripts/download_narratives_minimal.py --fetch-real`.
2. Set `NARRATIVES_ROOT` to the dataset root.
3. Train:

```powershell
$env:NARRATIVES_ROOT="C:\path\to\ds002345"
anima train --model distilgpt2 --narratives-root $env:NARRATIVES_ROOT --target-mode pca
```

Outputs `probes/zoo/distilgpt2_narratives_pca.pt`, optional `.calib.pt`, and `distilgpt2_tribe_proj.npz`.

Holdout stories are defined in `benchmarks/splits/narratives_holdout.json` (default: train `pieman`,`tunnel`; holdout `lucy`).

## Tiny default model (synthetic zoo)

```powershell
python scripts/build_zoo_tiny_probe.py
```

Creates a small trained checkpoint for `hf-internal-testing/tiny-random-gpt2` so the dashboard is not purely random.

## Validation

```powershell
anima validate --model distilgpt2 --prompt "Perhaps it might be unclear."
```
