# Anima: Dimensional LLM Readouts

## Technical Overview & Methodology

**Version:** 1.0 (beta)  
**Author:** Siddarth Boggarapu  
**Repository:** [github.com/Siddarthb07/Anima](https://github.com/Siddarthb07/Anima)  
**License:** MIT (code); Hugging Face model weights under their respective licenses

---

## 1. Introduction

**Anima** is open-source research instrumentation for Hugging Face **causal language models**. It records internal transformer activations during encoding or generation and maps them to three scalar **dimensional readouts**:


| Dimension       | Range    | Interpretation                                                  |
| --------------- | -------- | --------------------------------------------------------------- |
| **Valence**     | −1 to +1 | Pleasant ↔ unpleasant axis (Russell circumplex)                 |
| **Arousal**     | 0 to 1   | Activated ↔ calm axis                                           |
| **Uncertainty** | 0 to 1   | Model confidence / epistemic spread (probe + auxiliary signals) |


These numbers are **constructed measurements** from learned linear probes and logits diagnostics — not claims about machine consciousness, clinical affect, or subjective experience.

### 1.1 What Anima is

- A **measurement harness**: forward hooks → probe heads → REST/WebSocket API → optional live dashboard.
- A **training toolkit** for aligning probes with (a) text-emotion labels (GoEmotions) or (b) narrative fMRI time series (Narratives-style encoding models).
- A **benchmark runner** with manifest output for reproducibility.

### 1.2 What Anima is not

- Not a chat product, not an Ollama integration, not a feelings detector.
- Not clinical, diagnostic, or high-stakes decision tooling.
- Not a voxel-level fMRI decoder — the TRIBEv2 pathway is a **surrogate visualization** from the same activations.

**Approved language:** *readout*, *internal geometry*, *population-level analogy*.  
**Avoid:** *"the model feels anxious/happy"* stated as fact.

---

## 2. Problem Statement & Design Goals

Large language models produce fluent text, but text alone is an incomplete window into internal computation. Researchers studying interpretability often want **continuous, per-token signals** that track how internal representations evolve during generation — especially along affective and epistemic axes that correlate (imperfectly) with human annotation and, optionally, neural data.

Anima separates three objects that are often conflated:

1. **Observed text** — the tokens the user reads.
2. **Hidden activations** — residual-stream tensors inside the transformer.
3. **Brain BOLD signals** — slow, noisy population-level neural measurements (optional training target).

The system is designed to bridge (1) and (2) at inference time, and optionally align (2) with (3) during probe training using encoding-model methodology inspired by narrative fMRI literature.

### Design principles


| Principle                    | Implementation                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| **Honest scope**             | Random probes work for plumbing; scientific claims require trained checkpoints        |
| **Per-token resolution**     | Hooks fire on every forward pass during autoregressive generation                     |
| **Multi-signal uncertainty** | Probe head + entropy + logit gap + attention entropy, fused and optionally calibrated |
| **Layer transparency**       | Softmax-weighted fusion across probed layers; early/late disagreement detection       |
| **Reproducibility**          | CLI, benchmarks, manifest JSON, probe zoo with metadata sidecars                      |


---

## 3. System Architecture

### 3.1 High-level data flow

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│ User prompt │────▶│ ActivationExtractor  │────▶│ Forward hooks   │
│  (text)     │     │ (HF causal LM)       │     │ on N layers     │
└─────────────┘     └──────────────────────┘     └────────┬────────┘
                                                          │
                        ┌─────────────────────────────────┼──────────────────────────┐
                        ▼                                 ▼                          ▼
                 ┌─────────────┐                  ┌──────────────┐           ┌──────────────┐
                 │ AffectProbe │                  │ Uncertainty  │           │ TRIBEv2      │
                 │ (3 heads)   │                  │ from logits  │           │ surrogate    │
                 └──────┬──────┘                  │ + attention  │           │ (optional)   │
                        │                          └──────┬───────┘           └──────┬───────┘
                        └──────────────┬───────────────────┘                          │
                                       ▼                                                │
                              ┌─────────────────┐                                       │
                              │ Guard + region  │◀──────────────────────────────────────┘
                              │ + suppression   │
                              └────────┬────────┘
                                       ▼
                    ┌──────────────────────────────────────┐
                    │ FastAPI: /encode, /generate, WS stream │
                    └──────────────────┬───────────────────┘
                                       ▼
                    ┌──────────────────────────────────────┐
                    │ React dashboard (circumplex, bars)   │
                    └──────────────────────────────────────┘
```

### 3.2 Component map


| Layer         | Technology               | Role                                                                             |
| ------------- | ------------------------ | -------------------------------------------------------------------------------- |
| **Runtime**   | PyTorch + `transformers` | Load HF causal LM, register hooks, run encode/generate                           |
| **Core**      | `core/`                  | Layer config, hooks, extraction, region labels, guard, suppression               |
| **Probes**    | `probes/`                | `AffectProbe` network, training, zoo I/O, calibration                            |
| **Alignment** | `alignment/`             | Narratives loader, word→token align, TR binning, ridge encoding, TRIBE surrogate |
| **API**       | FastAPI + Uvicorn        | REST + WebSocket on port 8010                                                    |
| **Dashboard** | Vite + React + Recharts  | Live circumplex plot, uncertainty bars, token stream                             |
| **CLI**       | `anima` entry point      | `api`, `train`, `train-text`, `train-zoo`, `benchmark`, `bootstrap`              |


### 3.3 Model registry vs probe weights

Two configurations are kept separate:

- `**core/layer_config.py`** — which transformer layers to hook, `hidden_dim`, GPU/gated flags (shipped in git).
- `**probes/zoo/*.pt`** — trained probe weights (binary checkpoints gitignored; `.meta.json` sidecars in git).

Checkpoint resolution order (`probes/zoo_io.py`):

1. `{slug}_narratives_pca.pt` — brain-aligned (preferred)
2. `{slug}_text.pt` — GoEmotions text training
3. `{slug}.pt` — base fallback

If no checkpoint exists, the probe is **randomly initialized**. The pipeline runs end-to-end, but readouts are uncalibrated.

### 3.4 Supported models

**CPU tier:** `hf-internal-testing/tiny-random-gpt2`, `distilgpt2`, `TinyLlama-1.1B`, `Qwen2.5-0.5B`, `SmolLM2-1.7B`  
**GPU tier (7B+):** `Llama-3-8B`, `Mistral-7B`, `Qwen2-7B`, `Gemma-9B` (some gated, require `HF_TOKEN`)

Anima does **not** integrate with Ollama. Use the equivalent Hugging Face model id.

---

## 4. Measurement Methodology

This section describes *how* readouts are computed — the core scientific/engineering contribution.

### 4.1 Activation capture via forward hooks

`core/hooks.py` registers PyTorch forward hooks on selected transformer blocks. On each forward pass:

1. The hook captures the **residual stream** tensor `[batch, seq_len, hidden_dim]`.
2. Buffers are cleared before each pass to avoid stale activations.
3. Hooks are removed on API shutdown (`lifespan` cleanup in `api/server.py`).

`core/extractor.py` — `ActivationExtractor` exposes two paths:


| Path         | Method                  | Use case                                                           |
| ------------ | ----------------------- | ------------------------------------------------------------------ |
| **Encode**   | `encode_sequence(text)` | Single forward pass; one readout per input token                   |
| **Generate** | `extract_iter(prompt)`  | Autoregressive decoding with KV cache; one readout per *new* token |


### 4.2 The AffectProbe network

`probes/linear_probe.py` — `AffectProbe` is a small `nn.Module`:

**Step 1 — Layer fusion.** Activations from `N` probed layers are stacked and combined via **learned softmax weights**:

```
weights = softmax(layer_weights)
fused = Σᵢ weights[i] · activation[layer_i]
```

**Step 2 — Three linear heads** on the fused vector:


| Head          | Activation | Output range |
| ------------- | ---------- | ------------ |
| `valence`     | `tanh`     | [−1, +1]     |
| `arousal`     | `sigmoid`  | [0, 1]       |
| `uncertainty` | `sigmoid`  | [0, 1]       |


`heads_from_hidden()` applies the same head weights to a single layer's hidden state — used for suppression detection (early vs late layer comparison).

### 4.3 Auxiliary uncertainty signals

Independent of the probe's uncertainty head, `core/extractor.py` computes three signals from model outputs each token:


| Signal                | Definition                                                       | Intuition                             |
| --------------------- | ---------------------------------------------------------------- | ------------------------------------- |
| **Entropy**           | Normalized softmax entropy: `H / log(V)`                         | Flat distribution → high uncertainty  |
| **Logit gap**         | `1 / (1 + max(top1 − top2, 0))`                                  | Close top-2 logits → high uncertainty |
| **Attention entropy** | Mean head entropy of last-layer attention (final query position) | Diffuse attention → high uncertainty  |


**Fused uncertainty:**

```
fused = 0.35 · entropy + 0.35 · logit_gap + 0.30 · attn_entropy
```

The fused signal supervises brain-path probe training and feeds the guard policy. Optional **Platt scaling** (`probes/calibration.py`) calibrates fused scores on held-out data.

### 4.4 Region labeling (Russell circumplex)

`core/regions.py` — `label_region(valence, arousal, uncertainty)` maps scalars to named quadrants:

- If `uncertainty > 0.75` → `"high-uncertainty"` (no human emotion analog assigned).
- Otherwise, 2×2 valence × arousal grid: excitement, calm, anxiety, sadness, neutral.
- `region_analog` strings include mandated psychology disclaimers.

Confidence tiers (HIGH / MEDIUM / LOW) derive from fused uncertainty thresholds.

### 4.5 Suppression detection (layer disagreement)

`core/suppression.py` compares early-layer vs late-layer readouts using the same probe heads:


| Flag                    | Condition                                    | Framing                                 |
| ----------------------- | -------------------------------------------- | --------------------------------------- |
| `valence_suppression`   | late valence − early valence > 0.35          | Internal inconsistency, not "deception" |
| `uncertainty_overclaim` | late uncertainty − early uncertainty < −0.30 | Late layers under-report spread         |


Enabled by default on `/generate` and WebSocket streams (`detect_suppression: true`).

### 4.6 Guard policy (reliability gating)

`core/guard.py` — `evaluate_guard()` combines signals before surfacing a readout:

```
composite = 0.55 · fused_calibrated + 0.45 · probe_uncertainty
```

**Abstain** if ≥2 reasons fire or fused ≥ threshold. Reasons include:

- `high_fused_uncertainty`
- `high_probe_uncertainty`
- `lexical_hedging` — hedge-word count from `probes/validate.py` ("maybe", "perhaps", "I think", etc.)

When abstaining, the guard can override the region label to high-uncertainty.

### 4.7 TRIBEv2 surrogate pathway

`alignment/tribe_encoder.py` maps mean probed-layer activations to five named ROI-like axes via tanh-normalized dot products:

`tpj`, `amygdala`, `acc`, `vmpfc`, `broca`

Derived valence/arousal sketch:

```
valence ≈ vmpfc − amygdala
arousal ≈ acc
```

Weights are either seeded random or trained ridge projections saved as `{slug}_tribe_proj.npz`. This block exists for **dashboard visualization** — it is not voxel-level fMRI decoding and should not be cited as neuroscience evidence.

---

## 5. Training Pipelines

Probes must be trained before readouts carry semantic meaning. Anima ships two training paths plus a zoo builder.

### 5.1 Text-emotion training (GoEmotions)

**Entry:** `anima train-text --model <hf_id>`  
**Module:** `probes/train_text.py`


| Step          | Detail                                                                         |
| ------------- | ------------------------------------------------------------------------------ |
| Dataset       | GoEmotions (`google-research-datasets/go_emotions`)                            |
| Label mapping | `probes/emotion_va_map.py` — Russell circumplex coordinates per emotion        |
| Multi-label   | Mean valence/arousal across active labels; uncertainty scales with label count |
| Features      | Last-token activation of encoded text (max_length 64)                          |
| Loss          | MSE on valence, arousal, uncertainty                                           |
| Split         | 85/15 train/val                                                                |
| Output        | `probes/zoo/{slug}_text.pt` + `.meta.json`                                     |


### 5.2 Brain-alignment training (Narratives-style)

**Entry:** `anima train --model <hf_id> --narratives-root <path>`  
**Module:** `probes/train.py`


| Step       | Detail                                                                                                                      |
| ---------- | --------------------------------------------------------------------------------------------------------------------------- |
| Data       | OpenNeuro ds002345 (full) or synthetic minimal corpus in `data/narratives_minimal/`                                         |
| Alignment  | Word→token mapping via HF offset mappings (`alignment/word_token_align.py`)                                                 |
| Encoding   | Per-token activations + fused uncertainty during story encode                                                               |
| TR binning | Average hidden states per 2s TR from word onsets (`alignment/encoding_pipeline.py`)                                         |
| HRF lag    | Shift fMRI by 3 TRs                                                                                                         |
| Confounds  | Partial out word rate + drift (`alignment/confound_control.py`)                                                             |
| Targets    | **PCA mode** (default): 2-component PCA on training BOLD → valence/arousal proxies; or **atlas mode** with user ROI indices |
| Holdout    | Stories in `benchmarks/splits/narratives_holdout.json` (train: pieman, tunnel; holdout: lucy)                               |
| Outputs    | `{slug}_narratives_pca.pt`, `.calib.pt`, `{slug}_tribe_proj.npz`                                                            |


### 5.3 Probe zoo builder

```bash
anima train-zoo --tier cpu                    # TinyLlama, Qwen-0.5B, SmolLM2
ANIMA_TRAIN_LARGE=1 anima train-zoo --tier large   # 7B+ models (GPU)
python scripts/download_zoo.py                  # fetch Release v1.1.0 weights
```

CI trains CPU/GPU tiers via `.github/workflows/train-zoo.yml`; artifacts download into `probes/zoo/`.

### 5.4 Probe origin metadata


| `probe_origin`                      | Meaning                    |
| ----------------------------------- | -------------------------- |
| `random`                            | No checkpoint found        |
| `text_emotion`                      | GoEmotions training        |
| `narratives_fMRI`                   | Real ds002345              |
| `narratives_fMRI_synthetic_minimal` | Bundled minimal corpus     |
| `synthetic_tiny`                    | `build_zoo_tiny_probe.py`  |
| `zoo`                               | Loaded from GitHub Release |


---

## 6. API, Dashboard & Evaluation

### 6.1 REST endpoints


| Method | Path           | Purpose                                                           |
| ------ | -------------- | ----------------------------------------------------------------- |
| `GET`  | `/health`      | `{"status": "ok", "version": "1.0.0"}`                            |
| `GET`  | `/models`      | Supported models + zoo checkpoint status                          |
| `POST` | `/encode`      | Single forward pass; per-token readouts                           |
| `POST` | `/generate`    | Autoregressive generation; batch readouts + suppression           |
| `WS`   | `/ws/generate` | Stream one `kind: "token"` message per token, then `kind: "done"` |


Interactive docs: `http://127.0.0.1:8010/docs`

### 6.2 Per-token response (`AffectReadout`)

Each token returns:

```json
{
  "token": "...",
  "affect": { "valence": 0.12, "arousal": 0.45, "uncertainty": 0.31 },
  "region": "calm",
  "region_analog": "...",
  "flags": { "high_uncertainty": false, "likely_hedging": false },
  "confidence_tier": "MEDIUM",
  "uncertainty_signals": { "entropy": 0.4, "logit_gap": 0.3, "attn_entropy": 0.2, "fused": 0.31 },
  "tribe_v2": { "roi_scores": {...}, "derived_va": {...} },
  "guard": { "tier": "LOW", "abstain_recommended": false, "reasons": [] }
}
```

### 6.3 Dashboard

Vite + React app (`dashboard/`) on port 5173:

- **Circumplex plot** — valence × arousal trajectory during generation
- **Uncertainty bars** — probe vs fused vs component breakdown
- **Token stream** — live text with per-token color coding
- **Brain alignment panel** — TRIBEv2 surrogate ROI bars (labeled as sketch)

WebSocket client: `dashboard/src/hooks/useAffectStream.js`

### 6.4 Benchmark results (published)

`**distilgpt2` — live run 2026-05-24:**


| Benchmark               | Metric                        | Value         |
| ----------------------- | ----------------------------- | ------------- |
| Narratives holdout      | Val MSE                       | 0.081         |
|                         | Pearson r (valence / arousal) | 0.284 / 0.004 |
|                         | Word-rate baseline r          | 0.097         |
| GoEmotions (text probe) | Pearson r (valence / arousal) | 0.057 / 0.021 |
| Guard fixtures (n=4)    | Abstain accuracy / AUROC      | 1.00 / 1.00   |


Reproduce: `anima benchmark --model distilgpt2 --tiers internal,external,external_text,external_guard`

**Caveats:** Narratives numbers use the **synthetic minimal** corpus, not full ds002345 (~100 GB). GoEmotions benchmark uses ≤200 validation samples. Guard AUROC on 4 fixtures is smoke-test only.

---

## 7. Limitations, Ethics & Roadmap

### 7.1 Scientific limitations

- **Uncalibrated by default** — without `probes/zoo/*.pt`, probes are random; region labels rarely fire meaningfully.
- **Weak correlations** — even trained distilgpt2 probes show modest GoEmotions alignment (r ≈ 0.06 valence).
- **Synthetic fMRI** — default brain training uses minimal synthetic Narratives, not real ds002345.
- **TRIBEv2 is a sketch** — not real neural decoding; do not cite as brain imaging.
- **Tiny default model** — `hf-internal-testing/tiny-random-gpt2` produces nonsense text; for CI/plumbing only.

### 7.2 Ethical boundaries


| Do                                               | Don't                                            |
| ------------------------------------------------ | ------------------------------------------------ |
| Research, teaching, interpretability prototyping | Clinical diagnosis or treatment                  |
| Careful "readout" language in papers             | "The model feels X" as fact                      |
| Train and publish your own checkpoints           | High-stakes automation (hiring, law enforcement) |
| Label surrogate ROI as visualization             | Sell ROI plots as real neuroscience              |


### 7.3 Roadmap (from `docs/BUILD_PLAN.md`)


| Phase | Scope                                         | Status                                |
| ----- | --------------------------------------------- | ------------------------------------- |
| v1.0  | Hooks, probes, API, dashboard, benchmarks, CI | **Shipped**                           |
| v1.1  | Release probe assets + `download_zoo.py`      | **Partial** (script + Release v1.1.0) |
| v1.2  | Expanded guard fixtures, real Narratives tier | Planned                               |
| v2.0  | HF Space / Gradio demo, SAE integration flags | Planned                               |


---

## 8. Quick Reference

### Install & run

```bash
git clone https://github.com/Siddarthb07/Anima.git && cd Anima
python scripts/bootstrap.py
anima api --port 8010
cd dashboard && npm install && npm run dev   # http://127.0.0.1:5173
```

### Key files


| File                             | Purpose                                                |
| -------------------------------- | ------------------------------------------------------ |
| `core/extractor.py`              | Model load, hooks, encode/generate, uncertainty fusion |
| `probes/linear_probe.py`         | `AffectProbe` architecture                             |
| `probes/train.py`                | Brain-alignment training                               |
| `probes/train_text.py`           | GoEmotions text training                               |
| `api/server.py`                  | FastAPI app and readout assembly                       |
| `core/layer_config.py`           | Supported models and layer indices                     |
| `alignment/encoding_pipeline.py` | TR alignment, HRF lag, ridge encoding                  |


### 30-second pitch

> Anima hooks into a Hugging Face causal LM, captures hidden states at selected layers, and maps them to valence, arousal, and uncertainty readouts per token. Probes are trainable linear heads — optionally aligned to text emotions or narrative fMRI. It exposes a streaming API and dashboard for research instrumentation, not anthropomorphic claims.

---

*For usage boundaries see [USAGE_AND_LIMITATIONS.md](USAGE_AND_LIMITATIONS.md). For training commands see [TRAINING.md](TRAINING.md). For architecture details see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).*