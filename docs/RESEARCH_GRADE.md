# Research-grade criteria (Part A)

Anima is **research-grade** when every row below is satisfied. This is the living checklist for releases and college-app claims.

**Version map** (must match [BUILD_PLAN.md](BUILD_PLAN.md)):

| Tag | Tier |
|-----|------|
| **v1.1.0** | Dev / **synthetic brain** — `probe_origin: narratives_fMRI_synthetic_minimal` |
| **v1.2.0** | **Adoption** — public demo, CI benchmark smoke, polished API/dashboard (brain data still synthetic unless noted) |
| **v2.0.0** | **Real brain** — documented OpenNeuro [ds002345](https://openneuro.org/datasets/ds002345) subset (`probe_origin: narratives_fMRI`) |

**Current shipped tier (v1.1.0):** synthetic brain on `data/narratives_minimal/`. Usable for pipeline and dashboard QA; **not** a claim about real human fMRI.

**Next adoption milestone (v1.2.0):** portfolio-ready demo + trustworthy benchmark reporting — not real fMRI.

**Research milestone (v2.0.0):** real ds002345 holdout numbers (R6–R7 below).

| ID | Criterion | Status (2026-06-03) |
|----|-----------|---------------------|
| R1 | Data provenance documented (`docs/BRAIN_PROBE_DATA.md`) | Partial — synthetic documented; real TBD |
| R2 | Leakage-safe holdout (`benchmarks/splits/narratives_holdout.json`; train/holdout in meta) | **Done** |
| R3 | Baselines in meta + README comparison | **Done** (synthetic metrics) |
| R4 | GitHub Release + `download_zoo.py` + versioned manifests | **Done** (v1.1.0); manifest schema v1 |
| R5 | Limitations linked from README; `/models` shows `brain_data_tier` | **Done** |
| R6 | distilgpt2 + one CPU proxy on **real** fMRI | **Not done** (synthetic only) — target **v2.0.0** |
| R7 | Optional one 7B–8B real brain probe | Not started — target **v2.0.0** |
| R8 | distilgpt2 text probe quality (val + benchmark r) | **Not done** — retrain per BUILD_PLAN Phase 1 |
| R9 | Guard benchmarks n>50 before citing AUROC | **Not done** — BUILD_PLAN Phase 3.1 |

## Holdout split (R2)

Defined in [`benchmarks/splits/narratives_holdout.json`](../benchmarks/splits/narratives_holdout.json):

- **Train stories:** `pieman`, `tunnel`
- **Holdout:** `lucy`

Training uses `probes.train.load_narratives_split()` by default.

## Manifest schema

Benchmark manifests include `manifest_schema_version: 1`. Per-model aliases:

- `benchmarks/reports/latest_manifest.json` (last benchmark run)
- `benchmarks/reports/latest_<slug>_manifest.json` (e.g. `latest_distilgpt2_manifest.json`)

Commit alias files only — not every timestamped run folder. Prefer repo-relative paths in JSON.

## Hardware (summary)

| Resource | Synthetic tier (v1.1) | Real ds002345 subset (v2.0) |
|----------|----------------------|----------------------------|
| Disk | ~5 GB | **~80 GB free** recommended |
| RAM | 8–12 GB peak train | 10–14 GB; page file 16–32 GB on Windows 16 GB |
| GPU | Optional | Required for 7B only |

Full tables: [TRAIN_ON_YOUR_MACHINE.md](TRAIN_ON_YOUR_MACHINE.md).

## Gate script

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stress_v1.ps1
```

## Next steps toward v1.2 (adoption)

1. Phase 1 — retrain distilgpt2 text probe (`--max-samples 1500`); meet quality gates in BUILD_PLAN.
2. Phase 3 — expand guard fixtures (n>50); enable CI benchmark-smoke on push.
3. Phase 4 — deploy HF Space (tiny default); dashboard model selector.

## Next steps toward v2.0 (real brain, R6)

1. Download ds002345 subset → `NARRATIVES_ROOT` (see [`BRAIN_PROBE_DATA.md`](BRAIN_PROBE_DATA.md)).
2. `anima train --model distilgpt2 --narratives-root %NARRATIVES_ROOT%`
3. Re-benchmark holdout + Release **v2.0.0** with `probe_origin: narratives_fMRI`.
