# Anima v2 — Release notes

## v2.1.0 — Public demo release

**Release date:** 2026-07-17  
**Previous stable tag:** [v2.0.0](https://github.com/Siddarthb07/Anima/releases/tag/v2.0.0)

v2.1.0 is the adoption release: Anima now runs as a public, zero-install demo on Hugging Face Spaces, with the full React dashboard, FastAPI backend, and streaming readouts hosted on ZeroGPU hardware. No probe weights or readout semantics changed — v2.0.0 checkpoints remain current.

### Highlights

- **Live public demo:** [huggingface.co/spaces/sidb078/Anima](https://huggingface.co/spaces/sidb078/Anima) — the original dashboard (circumplex, token stream, uncertainty decomposition, stability, layer disagreement, fMRI surrogate sketch) served from a single Space.
- **Space architecture** (`space/app.py`): FastAPI hosts the built dashboard and API routes; Gradio is mounted under `/gradio` so ZeroGPU detects the `@spaces.GPU` entrypoint. Launch sequencing is hardened for ZeroGPU (blocking launch, SSR disabled, uvicorn fallback).
- **Demo models:** Qwen2.5-0.5B-Instruct (default Space hero), TinyLlama-1.1B-Chat (council best, 94.0), tiny-random-gpt2 (plumbing tier).
- **Public-mode security:** `ANIMA_PUBLIC_MODE` request limits, model allowlist, API-key support, and path sanitization (`core/limits.py`); public demo prefers text checkpoints.
- **Deploy pipeline:** `scripts/deploy_hf_space.py` + `deploy-hf-space` workflow (fails fast on missing `HF_TOKEN`).
- **Benchmarks/docs:** refreshed TinyLlama manifest and POC demo report; README leads with the TinyLlama hero path and live Space link.
- **Dependencies:** dashboard on React 19 + Tailwind 4 + recharts 3; CI actions bumped; `accelerate` added for `device_map` loading.

Readouts remain **instrumentation** — guard AUROC on fixtures is policy smoke, not hallucination detection, and brain probes stay on the synthetic-minimal tier (real ds002345 is v3.0.0+).

---

## v2.0.0

**Release date:** 2026-07-06  
**Previous stable tag:** [v1.1.0](https://github.com/Siddarthb07/Anima/releases/tag/v1.1.0)

Anima v2 is a production-oriented upgrade over v1: multi-model benchmark validation, trained text probes for CPU-tier instruct models, a richer dashboard, Docker stack profiles, rolling readout stability, and opt-in generation-time intervention. Readouts remain **instrumentation** — not claims of subjective experience. See [Usage & limitations](USAGE_AND_LIMITATIONS.md).

---

## Highlights

| Area | v1 | v2 |
|------|----|----|
| **Benchmarks** | Single-model manifests, smoke guard fixtures | Multi-model rollup, council scoring, validation rubric, chart pipeline |
| **Probes** | `distilgpt2`, `tiny-random-gpt2` Release weights | Text-probe meta for Qwen, TinyLlama, SmolLM2; refreshed distilgpt2 meta |
| **Dashboard** | Basic circumplex + uncertainty | Model selector, stability panel, layer disagreement, glossary, tribe surrogate |
| **Docker** | Single-service API image | Compose profiles (`pull` / `stack`), dashboard nginx proxy, `docker-up`/`docker-down` helpers |
| **API** | Token affect + guard | Rolling **stability score**, `guard_mode: gate`, `intervention_mode: dampen` |
| **Validation** | Ad-hoc README tables | Four-dimension rubric (schema, probe signal, honesty flags, prompt separation) |

---

## Benchmarks & validation rubric

### New capabilities

- **`benchmarks/council.py`** — weighted multi-judge council scores manifests and live prompt readouts (aggregate ≥60 = publication bar).
- **`scripts/run_all_models_benchmark.py`** — CPU-tier sweep across registered models in `core/layer_config.py`.
- **`scripts/generate_benchmark_report.py`** / **`scripts/generate_benchmark_charts.py`** — narrative report + PNG charts under `docs/images/benchmarks/`.
- **Expanded guard fixtures** — HaluEval and TruthfulQA guard samples scaled for CI smoke (policy behaviour only, not hallucination detection).
- **POC emotional prompts** — `benchmarks/fixtures/poc_emotional_prompts.json` for intervention demos.

### Validation rubric (four dimensions)

| Dimension | Weight | What it checks |
|-----------|--------|----------------|
| Schema integrity | 15% | Manifest schema, timestamps, complete entries |
| Probe signal strength | 35% | GoEmotions Pearson *r*, brain holdout *r*, smoke extract |
| Honesty flags | 20% | Penalises perfect AUROC on tiny fixtures, small *n* |
| Prompt separation | 30% | Positive vs negative live-prompt mean-valence gap |

Published rollup: [`benchmarks/reports/council_rollup.json`](../benchmarks/reports/council_rollup.json) (multi-model summary) · full report: [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) · live demo: [HF Space](https://huggingface.co/spaces/sidb078/Anima).

### CPU-tier results (2026-07-06)

| Model | Council | Passed | Notes |
|-------|---------|--------|-------|
| **TinyLlama/TinyLlama-1.1B-Chat-v1.0** | 94.0 | yes | **College-apps hero** — best prompt separation |
| **Qwen/Qwen2.5-0.5B-Instruct** | 91.0 | yes | Backup instruct demo |
| **distilgpt2** | 82.2 | yes | Strong live positive readouts; brain holdout *r* negative on synthetic tier |
| **SmolLM2-1.7B-Instruct** | 58.5 | no | Inverted prompt gap; valence *r* ≈ 0 — do not cite for validity |
| **tiny-random-gpt2** | 50.2 | no | CI/plumbing only |
| Llama-3.2-1B, gemma-2-2b-it | — | no | Gated HF repos (not run) |

Guard AUROC 1.0 across models is **fixture-policy smoke**, not production hallucination detection.

---

## Probes & zoo

- Retrained **distilgpt2** text probe (1500 GoEmotions samples) — updated `probes/zoo/distilgpt2_text.meta.json`.
- New text-probe metadata for **Qwen2.5-0.5B-Instruct**, **TinyLlama-1.1B-Chat**, **SmolLM2-1.7B-Instruct** (`probes/zoo/*_text.meta.json`).
- `scripts/download_zoo.py` and `scripts/train_text_zoo_all.py` extended for v2 CPU model list.
- Checkpoint `.pt` files ship via GitHub Release or local training — not in git (see `probes/zoo/README.md`).

---

## Dashboard

- **Model selector** — switch HF model id without restarting the dev server.
- **Stability panel** — rolling readout stability score and guard-abstain rate over the token stream.
- **Layer disagreement**, **tribe surrogate**, **glossary**, and **analysis caption** panels.
- **Dockerised dashboard** — `dashboard/Dockerfile`, nginx reverse proxy to API WebSocket/REST.
- `dashboard/src/apiBase.js` — configurable API base for compose vs local dev.

---

## Docker & deployment

- **`docker-compose.yml`** — `pull` profile (`scripts/pull_hf_models.py`) and `stack` profile (API + dashboard).
- **`scripts/docker-up.ps1`** / **`scripts/docker-down.ps1`** / **`scripts/docker-build.ps1`** — Windows helpers for model-specific stacks (`qwen`, `distil`, `tiny`).
- API Dockerfile: healthcheck, 8 GB memory limit, persistent HF cache volume.
- **`space/README.md`** — Hugging Face Spaces deploy notes for public demo.

---

## API & core

### Rolling stability (`core/stability.py`)

- Per-token stability score from a sliding window of valence/arousal swings and guard abstain rate.
- `guard_mode: gate` suppresses region labels when stability falls below threshold (`probes/guard_config.yaml`).

### Opt-in intervention (`core/intervention.py`)

- `intervention_mode: dampen` — experimental one-step residual correction opposite recent valence swing.
- Exposed on `POST /generate` and WebSocket generate; documented limits in [USAGE_AND_LIMITATIONS.md](USAGE_AND_LIMITATIONS.md).

### Other API changes

- Stability summary fields merged into generate response (`api/schemas.py`, `api/server.py`).
- `core/extractor.py` — dynamic int8 load path (`ANIMA_LOAD_DYNAMIC_INT8`), intervention hook integration.

---

## CLI & scripts

| Command / script | Purpose |
|------------------|---------|
| `anima benchmark --tiers internal,external,external_text,external_guard` | Unchanged entry; richer manifest schema |
| `python scripts/run_all_models_benchmark.py` | Full CPU-tier multi-model sweep |
| `python scripts/generate_benchmark_report.py` | Markdown report + chart regeneration |
| `python scripts/run_poc_demo.py` | POC intervention demo against emotional prompts |
| `python scripts/expand_guard_fixtures.py` | Regenerate expanded guard fixture JSON |

---

## Tests & CI

New test modules: `test_stability.py`, `test_intervention.py`, `test_council.py`, `test_manifest_paths.py`, `test_quantized_load.py`. CI runs benchmark-smoke on push (`.github/workflows/ci.yml`).

---

## Upgrade from v1.1.0

```bash
git pull
pip install -e ".[dev,bench]"
python scripts/download_zoo.py          # fetch or refresh probe weights
python scripts/download_narratives_minimal.py
```

Re-run benchmarks if you cite numbers in papers or README forks:

```bash
$env:NARRATIVES_ROOT=".\data\narratives_minimal"
$env:ANIMA_FORCE_CPU="1"
python scripts/run_all_models_benchmark.py
python scripts/generate_benchmark_report.py
```

Dashboard (local):

```bash
anima api --port 8010
cd dashboard && npm install && npm run dev
```

Docker:

```powershell
.\scripts\docker-up.ps1 qwen
# UI: http://localhost:8080  API: http://localhost:8010
```

---

## Known limits (unchanged philosophy)

- Readouts are **probes on hidden states**, not ground-truth emotion or neuroscience.
- Brain-aligned scores on `narratives_minimal` use **synthetic BOLD** — label as `synthetic_minimal`, not real fMRI.
- Guard metrics on expanded fixtures test **abstention policy**, not TruthfulQA/HaluEval leaderboard performance.
- `intervention_mode: dampen` is experimental research tooling, not a product safety layer.

---

## Documentation map

| Doc | Purpose |
|-----|---------|
| [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) | Full multi-model narrative + rubric notes |
| [BENCHMARK_PUBLISHING.md](BENCHMARK_PUBLISHING.md) | How to reproduce and publish benchmark updates |
| [USAGE_AND_LIMITATIONS.md](USAGE_AND_LIMITATIONS.md) | Required reading before demos or papers |
| [TRAIN_ON_YOUR_MACHINE.md](TRAIN_ON_YOUR_MACHINE.md) | CPU training, int8 load, one-model-at-a-time |

---

## Contributors

Built on the v1 open-source bootstrap (FastAPI + probes + Narratives alignment). v2 adds benchmark council, stability gating, intervention surface, and multi-model CPU validation.
