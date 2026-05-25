# Research-grade criteria (Part A)

Anima is **research-grade** when every row below is satisfied. This is the living checklist for releases and college-app claims.

**Current tier (v1.1.0):** **Dev / synthetic brain** — probes trained on `data/narratives_minimal/` (`probe_origin: narratives_fMRI_synthetic_minimal`). Usable for pipeline and dashboard QA; **not** a claim about real human fMRI.

**Target tier (v1.2.0):** **Real brain** — probes trained on an documented OpenNeuro [ds002345](https://openneuro.org/datasets/ds002345) subset (`probe_origin: narratives_fMRI`).

| ID | Criterion | Status (2026-05-25) |
|----|-----------|---------------------|
| R1 | Data provenance documented (`docs/BRAIN_PROBE_DATA.md`) | Partial — synthetic documented; real TBD |
| R2 | Leakage-safe holdout (`benchmarks/splits/narratives_holdout.json`; train/holdout in meta) | **Done** |
| R3 | Baselines in meta + README comparison | **Done** (synthetic metrics) |
| R4 | GitHub Release + `download_zoo.py` + versioned manifests | **Done** (v1.1.0); manifest schema v1 added |
| R5 | Limitations linked from README; `/models` shows `brain_data_tier` | **Done** |
| R6 | distilgpt2 + one CPU proxy on **real** fMRI | **Not done** (synthetic only) |
| R7 | Optional one 7B–8B real brain probe | Not started |

## Holdout split (R2)

Defined in [`benchmarks/splits/narratives_holdout.json`](../benchmarks/splits/narratives_holdout.json):

- **Train stories:** `pieman`, `tunnel`
- **Holdout:** `lucy`

Training uses `probes.train.load_narratives_split()` by default.

## Manifest schema

Benchmark manifests include `manifest_schema_version: 1`. Per-model aliases:

- `benchmarks/reports/latest_manifest.json` (last benchmark run)
- `benchmarks/reports/latest_<slug>_manifest.json` (e.g. `latest_distilgpt2_manifest.json`)

## Hardware (summary)

| Resource | Synthetic tier (today) | Real ds002345 subset (v1.2) |
|----------|------------------------|-----------------------------|
| Disk | ~5 GB | **~80 GB free** recommended |
| RAM | 8–12 GB peak train | 10–14 GB; page file 16–32 GB on Windows 16 GB |
| GPU | Optional | Required for 7B only |

Full tables: see internal planning doc or expand `docs/TRAIN_ON_YOUR_MACHINE.md`.

## Gate script

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stress_v1.ps1
```

## Next steps toward R6

1. Download ds002345 subset → `NARRATIVES_ROOT` (see [`BRAIN_PROBE_DATA.md`](BRAIN_PROBE_DATA.md)).
2. `anima train --model distilgpt2 --narratives-root %NARRATIVES_ROOT%`
3. Benchmark + Release **v1.2.0**.
