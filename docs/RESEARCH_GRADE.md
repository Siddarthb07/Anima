# Research-grade criteria (Part A)

Anima is **research-grade** when every row below is satisfied. This is the living checklist for releases and external claims.

**Version map** (must match [BUILD_PLAN.md](BUILD_PLAN.md)):

| Tag | Tier |
|-----|------|
| **v1.1.0** | Dev / **synthetic brain** — `probe_origin: narratives_fMRI_synthetic_minimal` |
| **v2.0.0** | **Shipped adoption + validation** — multi-model council benchmarks, CPU instruct text probes on Release, dashboard v2, Docker, stability/intervention API; **brain data still synthetic_minimal** |
| **v2.1.0** | **Public demo** — Gradio HF Space, TinyLlama hero ([live demo](https://huggingface.co/spaces/sidb078/Anima)) |
| **v3.0.0+** | Optional **real brain** — OpenNeuro [ds002345](https://openneuro.org/datasets/ds002345) holdout (`probe_origin: narratives_fMRI`); post-RD |

**Current shipped tier (v2.0.0):** synthetic brain on `data/narratives_minimal/` plus council-validated **text-emotion** probes. Usable for pipeline, dashboard, and portfolio demos; **not** a claim about real human fMRI.

**Next milestone (v2.1.0):** public demo URL + essay-ready hero narrative — still synthetic brain unless noted.

**Research milestone (v3.0.0+):** real ds002345 holdout numbers (R6–R7 below).

| ID | Criterion | Status (2026-07-12) |
|----|-----------|---------------------|
| R1 | Data provenance documented (`docs/BRAIN_PROBE_DATA.md`) | **Done** for synthetic; real tier documented as future |
| R2 | Leakage-safe holdout (`benchmarks/splits/narratives_holdout.json`; train/holdout in meta) | **Done** |
| R3 | Baselines in meta + README comparison | **Done** (synthetic metrics; negative holdout disclosed) |
| R4 | GitHub Release + `download_zoo.py` + versioned manifests | **Done** (v2.0.0); manifest schema v1 |
| R5 | Limitations linked from README; `/models` shows `brain_data_tier` | **Done** |
| R6 | distilgpt2 + one CPU proxy on **real** fMRI | **Not done** — target **v3.0.0+** (post-apps) |
| R7 | Optional one 7B–8B real brain probe | Not started — target **v3.0.0+** |
| R8 | distilgpt2 text probe quality (val + benchmark r) | **Done** — GoE validation r≈0.16; train holdout ≈0.18 |
| R9 | Guard benchmarks n>50 before citing AUROC | **Done** (n=52) — cite as **fixture-policy smoke only**, not hallucination detection |
| R10 | Council rubric + multi-model rollup for portfolio | **Done** — `benchmarks/reports/council_rollup.json` |
| R11 | Public demo with API bounds + model allowlist | **Done** — [HF Space](https://huggingface.co/spaces/sidb078/Anima) |

## What you may claim externally (today)

**Safe:**

- Multi-model **text-emotion probes** with GoEmotions validation metrics and council scores.
- Live **instrumentation** pipeline (hooks → probe → API → dashboard).
- **Honest limits** (synthetic brain tier, guard as policy smoke, readouts ≠ subjective experience).
- **TinyLlama** as hero model (council 94, strongest prompt separation).

**Not safe until v3.0.0+:**

- “Brain-aligned” or “fMRI-validated” without `probe_origin: narratives_fMRI` on real ds002345.
- Positive Narratives holdout for distilgpt2 (current synthetic holdout r ≈ **−0.39**).
- Guard AUROC as production hallucination detection.

## Holdout split (R2)

Defined in [`benchmarks/splits/narratives_holdout.json`](../benchmarks/splits/narratives_holdout.json):

- **Train stories:** `pieman`, `tunnel`
- **Holdout:** `lucy`

Training uses `probes.train.load_narratives_split()` by default.

## Manifest schema

Benchmark manifests include `manifest_schema_version: 1`. Per-model aliases:

- `benchmarks/reports/latest_manifest.json` (last benchmark run)
- `benchmarks/reports/latest_<slug>_manifest.json` (e.g. `latest_distilgpt2_manifest.json`)
- **Multi-model summary:** `benchmarks/reports/council_rollup.json` (preferred over `all_models_rollup.json`)

Commit alias files only — not every timestamped run folder. Prefer repo-relative paths in JSON.

## Hardware (summary)

| Resource | Synthetic tier (v2.0) | Real ds002345 subset (v3.0+) |
|----------|----------------------|----------------------------|
| Disk | ~5 GB | **~80 GB free** recommended |
| RAM | 8–12 GB peak train | 10–14 GB; page file 16–32 GB on Windows 16 GB |
| GPU | Optional | Required for 7B only |

Full tables: [TRAIN_ON_YOUR_MACHINE.md](TRAIN_ON_YOUR_MACHINE.md).

## Gate script

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stress_v1.ps1
```

## Next steps (v2.1.0 — public demo)

1. Pin [HF Space](https://huggingface.co/spaces/sidb078/Anima) on personal site metrics page.
2. Re-benchmark hero model on each Release refresh.

## Next steps (v3.0.0+ — optional research)

1. Download ds002345 subset → `NARRATIVES_ROOT` (see [`BRAIN_PROBE_DATA.md`](BRAIN_PROBE_DATA.md)).
2. `anima train --model distilgpt2 --narratives-root %NARRATIVES_ROOT%`
3. Re-benchmark holdout + tag **v3.0.0** with `probe_origin: narratives_fMRI`.
