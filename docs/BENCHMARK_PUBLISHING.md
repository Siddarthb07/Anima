# Publishing Anima benchmarks to GitHub

This guide describes how to run benchmarks, validate results with the **benchmark validation rubric**, generate charts, and land changes in a professional, reviewable commit sequence.

---

## 1. Prerequisites

```powershell
pip install -e ".[dev,bench]"
pip install matplotlib
python scripts/download_zoo.py
python scripts/download_narratives_minimal.py
$env:NARRATIVES_ROOT = ".\data\narratives_minimal"
$env:ANIMA_FORCE_CPU = "1"
```

Ensure text probes exist for models you intend to cite. Train missing probes **one model at a time** on CPU:

```powershell
python scripts/train_text_zoo_all.py --models HuggingFaceTB/SmolLM2-1.7B-Instruct --max-samples 500 --epochs 8
```

CPU tier list: `scripts/train_text_zoo_all.py` (`CPU_MODELS`).

---

## 2. Run the full benchmark suite

```powershell
python scripts/run_all_models_benchmark.py
```

Outputs:

| Artifact | Path |
|----------|------|
| Multi-model rollup | `benchmarks/reports/all_models_rollup.json` |
| Per-model manifests | `benchmarks/reports/latest_<slug>_manifest.json` |
| Run log | `benchmarks/reports/benchmark_run.log` |

Models on **GPU evaluation tier** (7B–9B) are listed in the rollup with status *deferred to GPU evaluation tier* — not omitted casually. Gated Hugging Face models appear as *requires Hugging Face access approval*.

---

## 3. Generate charts and reports

```powershell
python scripts/generate_benchmark_report.py
python scripts/generate_benchmark_charts.py
python scripts/benchmark_publish_review.py
```

| Deliverable | Path |
|-------------|------|
| Narrative report + tables | `docs/BENCHMARK_REPORT.md` |
| Publication review (README-safe) | `docs/BENCHMARK_PUBLISH_REVIEW.md` |
| Validation scores JSON | `benchmarks/reports/validation_rollup.json` |
| Chart PNGs | `docs/images/benchmarks/*.png` |

---

## 4. Benchmark validation rubric

Implemented in `benchmarks/council.py` (internal module). The publication review script exposes scores without informal naming.

| Dimension | Weight | Pass intent |
|-----------|--------|-------------|
| Manifest integrity | 15% | Reproducible manifest schema |
| Probe signal strength | 35% | GoEmotions r ≥ 0.15, smoke extract OK |
| Reporting honesty | 20% | Penalises perfect AUROC on tiny fixtures |
| Live prompt separation | 30% | Positive vs negative valence gap |

**Aggregate ≥ 60** with core dimensions passing → suitable for README citation with stated limits.

---

## 5. What to publish vs withhold

**Publish when:**

- `text_emotion` probe trained; `val_pearson_valence` meets gate in meta sidecar
- Live prompt separation shows positive mean valence > +0.2 for positive fixture
- Charts and `BENCHMARK_PUBLISH_REVIEW.md` agree

**State limits explicitly when:**

- Brain holdout Pearson r is negative on synthetic Narratives tier
- Negative prompts do not reach valence < −0.1 (weak negative separation)
- Guard AUROC is 1.0 on fixture rows (policy smoke, not production metric)
- Model was evaluated with random probe only

**Do not claim:**

- Hallucination detection from guard fixtures
- Subjective experience in the language model
- GPU-tier numbers without running on GPU hardware

---

## 6. Recommended commit map

Land changes in **small, reviewable commits** (do not squash unrelated work).

### Commit 1 — Text probes (weights only if releasing)

```
feat(probes): train GoEmotions text probes for TinyLlama and SmolLM2

- probes/zoo/tinyllama_1.1b_chat_v1.0_text.pt + .meta.json
- probes/zoo/smollm2_1.7b_instruct_text.pt + .meta.json
- probes/zoo/train_text_report.json
```

*Omit `.pt` from git if using GitHub Release assets; keep `.meta.json` in repo.*

### Commit 2 — Benchmark manifests

```
bench: refresh all-models rollup after text probe training

- benchmarks/reports/all_models_rollup.json
- benchmarks/reports/latest_*_manifest.json
- benchmarks/reports/benchmark_run.log
```

### Commit 3 — Charts

```
docs: add benchmark chart PNGs for CPU tier models

- docs/images/benchmarks/*.png
- docs/images/benchmarks/manifest.json
```

### Commit 4 — Reports and validation

```
docs: benchmark report and publication review (2026-07-06)

- docs/BENCHMARK_REPORT.md
- docs/BENCHMARK_PUBLISH_REVIEW.md
- benchmarks/reports/validation_rollup.json
```

### Commit 5 — Tooling

```
feat(scripts): benchmark publish review and chart generation

- scripts/benchmark_publish_review.py
- scripts/generate_benchmark_charts.py
- scripts/run_all_models_benchmark.py
- docs/BENCHMARK_PUBLISHING.md
```

### Commit 6 — README metrics table

```
docs(readme): update benchmark tables from latest validation rollup
```

### Commit 7 — Dashboard / Docker (if included in same release)

```
feat(dashboard): model dropdown with trained vs random probe groups

feat(docker): build-only stack profile and probe warmup
```

---

## 7. README integration checklist

- [ ] Embed `docs/images/benchmarks/benchmark_overview.png`
- [ ] Copy executive summary table from `BENCHMARK_PUBLISH_REVIEW.md`
- [ ] Link manifests with full paths
- [ ] Note evaluation tier: *CPU, synthetic Narratives minimal, 2026-07-06*
- [ ] Run `python -m pytest tests/test_health.py tests/test_models_api.py -q` before push

---

## 8. Release assets (optional)

For large `.pt` files, attach to GitHub Release and extend `scripts/download_zoo.py` asset list. Keep metric sidecars in the repository for verifiability.
