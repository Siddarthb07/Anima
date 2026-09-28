# Anima — feature inventory (deep)

**Repo HEAD at write time:** `b3c8154` (`main` = `origin/main`, v2.1.0 lineage)  
**Purpose:** Features people miss when skimming the README or asking an LLM for a bullet list. Numbers and file paths are from the code, not marketing.

> Readouts ≠ feelings. See [USAGE_AND_LIMITATIONS.md](USAGE_AND_LIMITATIONS.md).

---

## 1. Core readout pipeline

### Multi-layer probing (already shipped — not a roadmap item)

- Hooks attach to **several** transformer layers per model (`core/layer_config.py`), not one mid-layer.
  - Examples: `distilgpt2` → `[2, 4]`; TinyLlama / Qwen / SmolLM2 → three mid-depth indices; Llama-3-8B → `[8, 16, 24, 28]`.
- `AffectProbe` (`probes/linear_probe.py`) stacks those hidden vectors and fuses with **learned softmax layer weights**, then three linear heads:
  - valence → `tanh` ∈ [−1, 1]
  - arousal → `sigmoid` ∈ [0, 1] (some README blurbs say −1…1 — **code is 0…1**)
  - uncertainty → `sigmoid` ∈ [0, 1]
- Same head weights can be applied to a **single** layer via `heads_from_hidden()` — used for early/late disagreement, not a second probe.

### Two extraction paths (easy to conflate)

| Path | API | What it scores |
|------|-----|----------------|
| **Generate** | `POST /generate`, `WS /ws/generate` | Autoregressive **new** tokens (greedy argmax) |
| **Encode** | `POST /encode` | Every **input** token position in one forward pass |

Encode is what live guard evals use for Q+A claims. Generate is what the dashboard stream uses.

### Per-token uncertainty fusion (independent of the probe)

From logits + attentions (`core/extractor.py`), each step emits:

| Signal | Definition (approx.) |
|--------|----------------------|
| `entropy` | Softmax entropy of next-token logits, normalized by log(vocab) |
| `logit_gap` | Soft transform of top-1 vs top-2 logit gap (`1 / (1 + gap)`) |
| `attn_entropy` | Mean head entropy of last-layer attention at final query position |
| `fused` | **0.35·entropy + 0.35·logit_gap + 0.30·attn_entropy** |

Probe `uncertainty` and fused uncertainty are **different signals**. The guard mixes them.

### Chat-template alignment (easy to miss)

Instruct models with `use_chat_template: true` in layer config get `tokenizer.apply_chat_template(..., add_generation_prompt=True)` on both train and serve (`core/prompt_format.py`).  
Fallback: inject a no-op system message if the tokenizer rejects user-only roles. Base models (e.g. `distilgpt2`) pass text through unchanged. Training without this on chat models → distribution shift vs dashboard demos.

### Decoding detail

- Generation is **greedy** (`argmax`), not sampling — affects hedge-word incidence and stability.
- KV cache enabled on generate; `use_cache=False` on encode.
- Attention forced to `"eager"` so attention tensors exist for `attn_entropy`.
- Hooks store activations as **CPU float32** even if the LM is half/int8 (`core/hooks.py`).

---

## 2. Reliability stack (four mechanisms, not one “guard”)

People say “Anima has a guard.” Internally there are four cooperating pieces:

### A. Multi-signal abstain guard (`core/guard.py` + `probes/guard_config.yaml`)

Thresholds (defaults):

| Key | Default | Role |
|-----|---------|------|
| `fused_high` | 0.75 | (documented high band; composite tiers use 0.4 / 0.7 cuts) |
| `fused_abstain` | 0.82 | Single-signal abstain if fused alone is this high |
| `probe_uncertainty_abstain` | 0.78 | Probe-U reason |
| `hedge_words_threshold` | 2 | Lexical hedge count on **token text** |

Logic that skim-lists miss:

- Composite ≈ **0.55·fused_cal + 0.45·probe_u** (reweights when ablating signals).
- Abstain if **≥2 reasons** OR fused alone ≥ `fused_abstain`.
- Optional **Platt calibrator** on fused (`{slug}.calib.pt`) if present.
- `disabled_signals` ablations: `fused` | `probe_uncertainty` | `hedging` (`ABLATION_SIGNALS`).
- On abstain, region labels collapse to **high-uncertainty** (`region_under_guard`) — not just a boolean flag.

Lexical hedges (`probes/validate.py`): fixed set — `however`, `although`, `might`, `perhaps`, `possibly`, `unclear`, `uncertain`, `depends`, `maybe`, `could`. Count is on whitespace-split token text (generation tokens are often single pieces — hedge detection is stronger on encode spans / aggregated text).

### B. Rolling stability (`core/stability.py`)

| Key | Default |
|-----|---------|
| `stability_window` | 8 tokens |
| `stability_score_threshold` | 0.45 |
| `valence_swing_threshold` | 0.35 |
| `stability_abstain_rate` | 0.5 |

Score = `1 − (0.45·vol_penalty + 0.35·delta_penalty + 0.20·abstain_penalty)`.  
Per-token flag `unstable_window`; run-level `stability_score` in `summary`.

### C. Guard mode: observe vs gate

- `guard_mode=observe` (default): stability flags only.
- `guard_mode=gate`: unstable windows **overwrite** `region` / `region_analog` and set `flags.gated=true` (`apply_guard_gate`). Does not stop generation.

### D. Layer disagreement / “suppression” (`core/suppression.py`)

Uses **same** probe heads on earliest vs latest hooked layer:

| Event | Trigger |
|-------|---------|
| `valence_suppression` | late − early valence **> 0.35** |
| `uncertainty_overclaim` | late − early uncertainty **< −0.30** |

Severity HIGH if |Δv| > 0.5 or |Δu| > 0.45. These are **heuristic inconsistency flags**, not claims the model is “lying.” Dashboard toggle: `detect_suppression`.

### E. Opt-in residual intervention (`core/intervention.py`)

- `intervention_mode=dampen` **changes the residual stream** (and therefore the text), not just labels.
- Mid probed layer; direction = unit vector of `probe.valence.weight`.
- Alpha **0.12**; fires when consecutive token valence Δ ≥ **0.25**.
- Opposite swing direction (damps recent jump). Experimental — called out in usage limits.
- **Do not confuse with** `anima validate` steering: that path steers along the **uncertainty** head with α≈20 vs random controls. Production dampen = valence head; validate = uncertainty head.

CLI `anima validate --volatility` compares valence std with vs without dampen.

---

## 3. Circumplex / region labeling (metaphor, not neuroscience)

`core/regions.py` thresholds:

| Region | Rule |
|--------|------|
| high-uncertainty | uncertainty > 0.75 |
| high-positive-activation | v > 0.4 and a > 0.5 |
| low-positive-activation | v > 0.4 and a ≤ 0.5 |
| high-negative-activation | v ≤ −0.4 and a > 0.5 |
| low-negative-activation | v ≤ −0.4 and a ≤ 0.5 |
| neutral | else |

Flags: `high_uncertainty`, `negative_valence`, `high_arousal`, `likely_hedging` (u > 0.6 and v < 0).  
`confidence_tier` is from **fused** entropy blend (HIGH if fused < 0.4), not from probe-U alone — easy to misread in the UI.

---

## 4. Probe training & zoo

### Text path (primary)

- Dataset: GoEmotions simplified → hand-mapped Russell-style VA table (`probes/emotion_va_map.py`).
- Multi-label → mean V/A; uncertainty rises with label count: `min(0.95, 0.45 + 0.08·(n_labels−1))`.
- Checkpoint: `probes/zoo/{slug}_text.pt` + `.meta.json`; `probe_origin: text_emotion`.

### Brain / Narratives path

- Targets: PCA on BOLD (`pca`), optional atlas ROI indices (`atlas`), tribe_distill/blend CLI aliases that still train PCA then save tribe proj.
- Holdout stories fixed in `benchmarks/splits/narratives_holdout.json` (train: `pieman`, `tunnel`; holdout: `lucy`).
- Shipped tier today: **`narratives_fMRI_synthetic_minimal`** under `data/narratives_minimal/` — not OpenNeuro ds002345.
- Confound residualization + word-rate / position drift baselines live in `alignment/confound_control.py` + `encoding_pipeline.py` (encoding benchmarks, not the live API path).
- Subword→word alignment via HF offset mappings (`alignment/word_token_align.py`).

### Checkpoint resolution order (subtle)

Default: **brain (`_narratives_pca`) → text → bare slug**.  
If `ANIMA_PUBLIC_MODE=1` **or** `ANIMA_PREFER_TEXT_PROBE=1`: **text first**.  
Public Space always prefers text so demos don’t silently load synthetic-brain weights.

Missing checkpoint → **random** probe weights (wiring still works; semantics garbage). `/models` exposes `probe_origin`, `brain_data_tier` (`none` | `synthetic_minimal` | `real_fMRI`).

### Quantization (not TurboQuant)

| Env | Effect |
|-----|--------|
| `ANIMA_LOAD_8BIT=1` | bitsandbytes 8-bit on CUDA when `hidden_dim ≥ 2048` — intended for **7B train/serve on GPU** |
| `ANIMA_LOAD_DYNAMIC_INT8=1` | PyTorch dynamic int8 on CPU Linear layers — **inference**; train fp32 first |
| `ANIMA_FORCE_CPU=1` | Force CPU device map |
| `ANIMA_SLOW_TOKENIZER=1` | Slow tokenizer (RAM-friendly for text train; brain train wants fast) |
| `ANIMA_TRAIN_LARGE=1` | Unlocks large zoo tier scripts |

TurboQuant-style **KV** compression is **not** integrated and does not replace weight load for probe training.

---

## 5. TRIBEv2 surrogate (what it is / isn’t)

`alignment/tribe_encoder.py`:

- Five named axes: `tpj`, `amygdala`, `acc`, `vmpfc`, `broca`.
- Default: **seeded unit projections** from `sha256(model_name)` — deterministic, inspectable, **not** voxel TRIBE.
- Optional trained `*_tribe_proj.npz` flips mode to `surrogate_trained`.
- `ANIMA_TRIBE_MODE`: `surrogate` (default) | `runtime` | `blend` (`alignment/tribe_runtime.py`; optional `anima[tribe]`).
- Each token also gets `derived_va` from ROI scalars for the dashboard sketch.

---

## 6. API & deployment surface

### Endpoints

| Method | Path | Notes |
|--------|------|-------|
| GET | `/health` | Version / liveness |
| GET | `/models` | Zoo presence, metrics, brain tier |
| POST | `/encode` | Stimulus encoding |
| POST | `/generate` | Full generate + summary |
| WS | `/ws/generate` | Stream `token` / `done` / `error` messages |
| GET | `/` + SPA | Only if `ANIMA_SERVE_DASHBOARD` points at `dashboard/dist` |

Request knobs: `detect_suppression`, `guard_mode` (`observe`\|`gate`), `intervention_mode` (`none`\|`dampen`).

**Default mismatch (easy to miss):** API schema defaults `guard_mode=observe`; the React dashboard advanced controls default to **`gate`**. `/encode` summaries skip stability finalize; `/generate` and WS apply it.

### Public-mode safety (`core/limits.py`)

When `ANIMA_PUBLIC_MODE=1` (HF Space default):

- Model allowlist: tiny-random-gpt2, Qwen2.5-0.5B-Instruct, TinyLlama-1.1B-Chat.
- Caps: `ANIMA_MAX_NEW_TOKENS` (Space sets **64**), `ANIMA_MAX_PROMPT_CHARS` (8000), `ANIMA_MAX_ENCODE_LENGTH` (512).
- Optional `ANIMA_API_KEY` via header `X-Anima-API-Key`.
- Path sanitization on narratives roots; CORS via `ANIMA_CORS_ORIGINS`.

### Runtime ops details

- `ANIMA_WARMUP_MODEL` preloads one model at lifespan (Docker profiles set this).
- LRU-style eviction: loading a new model drops others (`_evict_other_models`) — important on 16 GB / SpaceGPU.
- Defaults: local default LM is **tiny-random-gpt2**; Space hero is **Qwen**; council-best remains **TinyLlama** (`core/defaults.py`).

---

## 7. Dashboard & demos

React/Vite dashboard (`dashboard/`):

- Live WebSocket token stream + REST fallback.
- Circumplex / token stream / uncertainty decomposition / stability panel / layer-disagreement panel.
- Toggles for suppression detection, guard mode, intervention mode.
- Model selector driven by `/models`.

Also:

- Gradio Space wrapper (`space/app.py`) mounts built dashboard + public env defaults.
- Docker compose profiles (`scripts/docker-up.ps1|sh` with e.g. `qwen`).
- Native Windows: `scripts/start_anima.ps1`.
- POC / dash demo Playwright helpers (`scripts/run_poc_demo.py`, `run_dash_demo.py`).

---

## 8. Benchmarks & honesty tooling

### Council (`benchmarks/council.py`)

Weighted judges — not a vibes score:

| Judge | Weight | What it checks |
|-------|--------|----------------|
| schema_integrity | 0.15 | Manifest schema v1, model id, timestamps |
| probe_signal | 0.35 | GoEmotions r gates (≥0.15 valence), brain holdout, smoke |
| honesty_flags | 0.20 | **Penalizes** perfect AUROC on tiny fixtures |
| prompt_separation | 0.30 | Live pos vs neg prompt valence gap |

Honesty judge explicitly docks “AUROC≈1 on n&lt;50 fixture smoke” — unusual for demo repos and easy for skim-lists to omit.

### Guard eval tiers (do not conflate)

1. **Fixture policy smoke** — pre-baked affect/uncertainty rows (n=52); AUROC 1.00 by construction when HIGH/LOW inputs are separable; good for CI plumbing; **not** hallucination detection. Do not put fixture AUROC in headline README tables.
2. **Live claim encode** (`benchmarks/live_guard_eval.py`) — real HaluEval / TruthfulQA-style pairs through extractor+probe; scores abstain vs labels; still **encode**, not free-gen + LLM judge. **This is the number to cite.**
3. **Ablations** (`benchmarks/run_guard_ablation.py`) — full / no_fused / no_probe / no_hedge / singles + optional semantic-entropy summary.

Live eval nuance: uncertainty aggregated over the **last third** of tokens (answer span), not the shared question prefix.

### Manifest hygiene

- `manifest_schema_version: 1`
- Per-model aliases `latest_<slug>_manifest.json` + `council_rollup.json`
- Prefer repo-relative paths; `*.pt` weights on GitHub Releases, not git

---

## 9. CLI (`anima …`)

| Command | Role |
|---------|------|
| `api` | uvicorn FastAPI |
| `smoke` | Tiny extract |
| `train` | Narratives brain probe (`--target-mode pca\|atlas\|tribe_distill\|blend`) |
| `train-text` | GoEmotions |
| `train-text-all` / `train-zoo` | Batch zoo tiers |
| `validate` | Hedge steering + optional `--volatility` |
| `benchmark` | Tiered suite |
| `bootstrap` | First-run data + tests |

Ollama is **unsupported** — HF ids only (`scripts/ollama_to_hf.json` is a map, not a runtime).

---

## 10. Model zoo quirks

- Gated families need `huggingface-cli login` / `HF_TOKEN` (Llama, Gemma).
- `has_sae` + `sae_release` flags exist for Llama-3-8B / Gemma-9B — **config-advertised only**; no SAE path in the live extractor.
- Optional `tribev2` runtime adapter and LITcoder “compare” are **stubs / naming shims**, not vendored upstreams.
- `requires_gpu` rows skipped on CPU zoo scripts unless `ANIMA_TRAIN_LARGE=1`.
- Zoo download pin: `ANIMA_ZOO_RELEASE` (default release tag in `scripts/download_zoo.py`).
- Space also mounts Gradio at `/gradio` only as a **ZeroGPU detector** (`@spaces.GPU`); the real UI is the FastAPI-served dashboard.

---

## 11. Env var cheat sheet

| Variable | Purpose |
|----------|---------|
| `ANIMA_FORCE_CPU` | CPU device map |
| `ANIMA_LOAD_8BIT` | bnb 8-bit (GPU, large hidden) |
| `ANIMA_LOAD_DYNAMIC_INT8` | CPU dynamic int8 inference |
| `ANIMA_SLOW_TOKENIZER` | Slow tokenizer for low-RAM text train |
| `ANIMA_TRAIN_LARGE` | Enable 7B+ zoo train scripts |
| `ANIMA_PREFER_TEXT_PROBE` | Prefer `*_text.pt` over brain |
| `ANIMA_PUBLIC_MODE` | Allowlist + text preference |
| `ANIMA_API_KEY` | Optional header auth |
| `ANIMA_MAX_NEW_TOKENS` / `_PROMPT_CHARS` / `_ENCODE_LENGTH` | Caps |
| `ANIMA_WARMUP_MODEL` | Lifespan preload |
| `ANIMA_SERVE_DASHBOARD` | Path to `dashboard/dist` |
| `ANIMA_CORS_ORIGINS` | CORS list |
| `ANIMA_TRIBE_MODE` | surrogate / runtime / blend |
| `ANIMA_ZOO_RELEASE` | Release tag for zoo download |
| `ANIMA_DEBUG` | Extra WS/error detail |
| `ANIMA_API_PORT` / `ANIMA_DASH_PORT` | Local ports |
| `ANIMA_HERO_MODEL` / `ANIMA_DEMO_*` | Demo scripts |

Legacy aliases: some scripts still accept `LLM_AFFECT_*`.

---

## 12. Explicit non-features (skim lists invent these)

- Not proof of subjective experience or clinical emotion AI.
- Not voxel-level TRIBE / individual fMRI decoding.
- Not Ollama runtime.
- Not sampling-based decoding in the main API path.
- Not automatic hallucination detection for production (fixture AUROC ≠ product claim).
- Not TurboQuant / exotic KV compression.
- Not multi-GPU serving / continuous batching.
- Brain probes on Release are **synthetic_minimal** until a real ds002345 holdout ships (v3.0.0+).
- Random probe if zoo weights missing — pipeline “works,” meaning is empty.

---

## 13. One-line architecture (accurate)

**Anima hooks selected layers of a Hugging Face causal LM, fuses multi-layer residuals through a small affect probe, attaches independent logit/attention uncertainty, runs multi-signal abstain + rolling stability + early/late disagreement (and optional residual dampen), and streams the result through FastAPI + a React dashboard — with text (GoEmotions) or Narratives-style brain supervision, honestly labeled.**

---

## Related docs

- [ANIMA_TECHNICAL_OVERVIEW.md](ANIMA_TECHNICAL_OVERVIEW.md) — methodology narrative  
- [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) — council tables and reproducibility  
- [BENCHMARKS.md](BENCHMARKS.md) — how to run / publish numbers  
- [RESEARCH_GRADE.md](RESEARCH_GRADE.md) — claim gates  
