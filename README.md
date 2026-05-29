# Anima

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Anima** is an open-source tool to **read emotion-style signals from large language models (LLMs)** as they generate text—**valence** (negative ↔ positive), **arousal** (calm ↔ intense), and an **uncertainty** score—using small neural **probes** on transformer hidden states. Think of it as a **live emotion meter for Hugging Face causal LMs**: hook layers → probe → numbers per token → API + dashboard.

> **Plain English:** Anima watches the model’s internal activations while it writes and outputs **emotion-like readouts** for each word/token. That helps you study **LLM affect**, **emotion probing**, and **interpretability**—not a chat app and not proof the model truly “feels” anything. See [Usage & limitations](docs/USAGE_AND_LIMITATIONS.md).

**Keywords:** LLM emotions · emotion readouts · valence/arousal probing · Hugging Face interpretability · fMRI-aligned brain probes (optional) · FastAPI streaming dashboard.

Anima does **not** use Ollama. Use a [supported Hugging Face model id](core/layer_config.py) (e.g. `distilgpt2`) with matching weights in `probes/zoo/`.

---

## What Anima does (concrete)

Given a prompt and a Hugging Face model id (e.g. `distilgpt2`):

1. **Load** `AutoModelForCausalLM` and attach **forward hooks** on layers listed in [`core/layer_config.py`](core/layer_config.py).
2. **Generate** tokens one at a time (or encode a fixed string with `/encode`).
3. For **each token**, read the hooked hidden vectors and run them through a small **probe network** ([`probes/linear_probe.py`](probes/linear_probe.py)) trained to predict **emotion dimensions**:
   - **valence** (−1 … 1) — how negative vs positive the readout looks
   - **arousal** (−1 … 1) — how calm vs intense it looks
   - **uncertainty** — how much to trust this token’s readout (not a medical score)
4. Add **diagnostics** from logits/attention (entropy-style signals, layer disagreement) in [`core/extractor.py`](core/extractor.py).
5. Optionally map the **same** activations through a **TRIBEv2 surrogate** ([`alignment/tribe_encoder.py`](alignment/tribe_encoder.py))—named ROI-like scalars for the UI. This is a **linear sketch for visualization**, not voxel-level fMRI decoding.
6. Apply a **guard** policy ([`core/guard.py`](core/guard.py)) that can recommend abstaining when readouts look unreliable (benchmarked on small fixtures).
7. Detect **suppression-style shifts** ([`core/suppression.py`](core/suppression.py)) when early vs late token readouts diverge sharply (heuristic inconsistency flag, not “lying”).

**Outputs:** JSON per token (affect, region label, flags, tribe surrogate, guard) via **REST** `POST /generate` or **WebSocket** streaming. Optional **React dashboard** plots valence/arousal over time.

```
  Prompt → HF causal LM → hooks (layers L₁…Lₖ)
                              ↓
                    AffectProbe (trained .pt)
                              ↓
              valence / arousal / uncertainty  +  guard + suppression events
                              ↓
                    FastAPI  →  dashboard (live)
```

---

## Two ways probes get their meaning

| Path | Training data | Checkpoint | `probe_origin` (typical) |
|------|----------------|------------|---------------------------|
| **Text** | [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) labels → valence/arousal mapping | `probes/zoo/{slug}_text.pt` | `text_emotion` |
| **Brain-aligned** | Story text + fMRI (Narratives layout; OpenNeuro [ds002345](https://openneuro.org/datasets/ds002345) or dev subset) | `probes/zoo/{slug}_narratives_pca.pt` | `narratives_fMRI` or `narratives_fMRI_synthetic_minimal` |

The API prefers the **brain** checkpoint when present, then text, then an uninitialized probe (**random** readouts—fine for wiring tests only).

**Published weights (CPU tier):** [GitHub Release v1.1.0](https://github.com/Siddarthb07/Anima/releases/tag/v1.1.0) — `distilgpt2` and `hf-internal-testing/tiny-random-gpt2`. Brain probes in v1.1.0 are trained on **synthetic minimal** BOLD ([`data/narratives_minimal/`](data/narratives_minimal/)), not full real fMRI. Details: [`docs/BRAIN_PROBE_DATA.md`](docs/BRAIN_PROBE_DATA.md).

```bash
python scripts/download_zoo.py    # fetch Release checkpoints into probes/zoo/
```

---

## What you get on each token

Example fields from `POST /generate` (see [`api/schemas.py`](api/schemas.py)):

| Field | Meaning |
|-------|---------|
| `affect.valence`, `affect.arousal`, `affect.uncertainty` | Probe head outputs |
| `region`, `region_analog` | Thresholded labels from readout geometry (metaphor, not neuroscience) |
| `flags` | e.g. high_uncertainty |
| `confidence_tier` | Coarse reliability bucket |
| `tribe_v2.roi_scores` | Surrogate ROI scalars (same activations as probe) |
| `guard.abstain_recommended` | Policy suggests not trusting this readout |
| `brain_alignment_note` | How probe was trained (`probe_origin` in summary) |

`GET /models` lists each supported HF id with `brain_data_tier` (`none` | `synthetic_minimal` | `real_fMRI`), holdout stories, and validation metrics when meta exists.

---

## Quick start

```bash
git clone https://github.com/Siddarthb07/Anima.git
cd Anima
pip install -e ".[dev]"
python scripts/download_zoo.py          # optional: Release probes
python scripts/bootstrap.py           # minimal data + tests
```

**Terminal 1 — API (port 8010):**

```bash
anima api --port 8010
# health: http://127.0.0.1:8010/health
```

**Terminal 2 — dashboard:**

```bash
cd dashboard && cp .env.example .env && npm install && npm run dev
# UI: http://127.0.0.1:5173  (proxies WebSocket to API)
```

Windows helper: `powershell -ExecutionPolicy Bypass -File scripts\start_anima.ps1`

**Smoke request:**

```bash
curl -X POST http://127.0.0.1:8010/generate \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"distilgpt2\",\"prompt\":\"Hello\",\"max_new_tokens\":8}"
```

Default model for low RAM: `hf-internal-testing/tiny-random-gpt2` (decoded text is intentionally noisy; pipeline still runs).

![Live emotion readouts on the dashboard](docs/images/dashboard-readout-example.png)

---

## Train your own probes

```bash
# Text probe (GoEmotions)
anima train-text --model distilgpt2 --max-samples 1500

# Brain probe (set NARRATIVES_ROOT to narratives_minimal or ds002345)
python scripts/download_narratives_minimal.py
anima train --model distilgpt2 --narratives-root ./data/narratives_minimal

# Benchmark holdout + text + guard tiers
anima benchmark --model distilgpt2 --tiers internal,external,external_text,external_guard
```

Holdout stories are fixed in [`benchmarks/splits/narratives_holdout.json`](benchmarks/splits/narratives_holdout.json) (train: `pieman`, `tunnel`; holdout: `lucy`). More commands: [`docs/TRAINING.md`](docs/TRAINING.md).

---

## Benchmarks — how well do the emotion readouts track real targets?

Anima ships a **benchmark suite** that scores your probes on public tasks. Each run writes a `manifest.json` you can cite or reproduce.

```bash
anima benchmark --model distilgpt2 --tiers internal,external,external_text,external_guard
```

### What each benchmark checks (simple)

| Benchmark | What it measures | In one sentence |
|-----------|------------------|-----------------|
| **Narratives holdout** | Brain-aligned probe vs story fMRI targets | “When the model reads a held-out story, do valence/arousal tracks match brain-derived targets better than guessing?” |
| **GoEmotions** | Text-emotion probe vs human emotion labels | “Do hidden states predict human-labeled emotion (mapped to valence/arousal) on tweet text?” |
| **HaluEval / TruthfulQA guard** | When to **not** trust a readout | “Does the guard flag unreliable emotion scores on tiny test fixtures?” |
| **Smoke extract** | Pipeline runs end-to-end | “Do hooks + probes return tokens without crashing?” |

**Holdout rule:** stories `pieman` + `tunnel` train, **`lucy` is held out** — see [`benchmarks/splits/narratives_holdout.json`](benchmarks/splits/narratives_holdout.json).

**Data honesty:** Narratives scores below use **`data/narratives_minimal/`** (synthetic fMRI for dev), **not** the full OpenNeuro ds002345 release yet. Label them as **synthetic_minimal** in papers. Real-fMRI tier: [`docs/BRAIN_PROBE_DATA.md`](docs/BRAIN_PROBE_DATA.md).

### Latest results (CPU tier)

#### `distilgpt2` — [full manifest](benchmarks/reports/latest_distilgpt2_manifest.json) (2026-05-24)

| Benchmark | Metric | Result | Beat simple baseline? |
|-----------|--------|--------|------------------------|
| **Narratives holdout** (`lucy`) | Pearson r (valence / arousal) | **0.28** / 0.004 | Valence **yes** vs word-rate r ≈ **0.10** |
| | Val MSE | 0.081 | — |
| **GoEmotions** (validation, ≤200 samples) | Pearson r (valence / arousal) | 0.06 / 0.02 | Weak; text probe still training-limited |
| **HaluEval guard** (n=4 smoke) | Abstain accuracy / AUROC | 1.00 / 1.00 | Fixture smoke only |
| **TruthfulQA guard** (n=4 smoke) | Abstain accuracy / AUROC | 1.00 / 1.00 | Fixture smoke only |
| **TRIBE reference** | Runtime decoder | skipped | Surrogate-only path in CI |
| **Brain-Score Language** | — | skipped | Install optional package |

#### `hf-internal-testing/tiny-random-gpt2` — [manifest](benchmarks/reports/latest_manifest.json) (dev / CI)

| Benchmark | Pearson r (valence / arousal) | Notes |
|-----------|-------------------------------|--------|
| Narratives holdout | **−0.11** / −0.24 | For plumbing only; LM output is random noise |
| GoEmotions | ~0.004 / ~0.01 | Not for emotion claims |

**How to read r:** closer to **1** = probe emotion tracks line up more with the target; **0** ≈ no linear relationship; negative = inverse trend (often means “not trained yet”).

**Reproduce:**

```bash
$env:NARRATIVES_ROOT=".\data\narratives_minimal"   # Windows
anima benchmark --model distilgpt2 --tiers internal,external,external_text,external_guard
```

More detail: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## What Anima is not

- A chatbot, therapy tool, or “emotion detector” for humans  
- Ollama / GGUF inference (use matching **Hugging Face** ids; see [`scripts/ollama_to_hf.json`](scripts/ollama_to_hf.json))  
- Proof of subjective experience in LMs  
- Real TRIBE fMRI decoding (surrogate block is labeled in API responses)

---

## Architecture (one screen)

| Component | Role |
|-----------|------|
| [`core/`](core/) | Layer map, hooks, streaming generation, suppression |
| [`probes/`](probes/) | `AffectProbe`, training, `probes/zoo/*.pt` |
| [`alignment/`](alignment/) | Narratives loader, word–token align, TRIBEv2 surrogate |
| [`api/`](api/) | FastAPI + WebSocket protocol |
| [`dashboard/`](dashboard/) | Vite/React live plots |
| [`benchmarks/`](benchmarks/) | Holdout runners + `manifest.json` reports |

Deeper walkthrough: [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md).

---

## CLI

```bash
anima api --port 8010
anima train-text --model <hf_id>
anima train --model <hf_id> --narratives-root <path>
anima train-zoo --tier cpu
anima benchmark --model <hf_id> --tiers internal,external,external_text,external_guard
```

---

## Documentation

| Doc | When to read |
|-----|----------------|
| [Getting started](docs/GETTING_STARTED.md) | Install, Docker, troubleshooting |
| [Researcher quickstart](docs/RESEARCHER_QUICKSTART.md) | Reproduce with Release weights in ~10 min |
| [Models & zoo](docs/MODELS_AND_ZOO.md) | HF ids, checkpoint naming, Ollama clarification |
| [Brain probe data](docs/BRAIN_PROBE_DATA.md) | Synthetic vs real ds002345 |
| [Research-grade criteria](docs/RESEARCH_GRADE.md) | What “research-grade” means here |
| [Usage & limitations](docs/USAGE_AND_LIMITATIONS.md) | **Before** papers, apps, or demos |
| [Training](docs/TRAINING.md) · [Benchmarks](docs/BENCHMARKS.md) | Commands and manifests |

---

## Development

```bash
python -m pytest -q -k "not distilgpt2"
powershell -ExecutionPolicy Bypass -File scripts\stress_v1.ps1
```

CI: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

---

## License

[MIT](LICENSE). Hugging Face **model weights** and **datasets** (GoEmotions, Narratives, etc.) have their own terms—you are responsible for compliance.
