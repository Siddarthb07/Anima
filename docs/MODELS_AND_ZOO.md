# Models, probe zoo, and Ollama

This page is for **anyone** using or packaging Anima — not a single deployment.

## Anima does not run Ollama

Anima loads **Hugging Face causal LMs** via `transformers` (`AutoModelForCausalLM`) and registers **forward hooks** on hidden states. There is **no** Ollama API integration, no local GGUF path, and no probe weights keyed by Ollama model names.

If you use **Ollama** for chat elsewhere, that stack is separate. To use Anima on a similar *family* of models, load the **same architecture** from Hugging Face (same weights family, different packaging), for example:

Full mapping file: [`scripts/ollama_to_hf.json`](../scripts/ollama_to_hf.json)

| Ollama-style name (informal) | Hugging Face id | CPU proxy to train without 7B RAM |
|-----------------------------|-----------------|-----------------------------------|
| llama3 / llama3.1 8B        | `meta-llama/Meta-Llama-3-8B-Instruct` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| mistral 7B                  | `mistralai/Mistral-7B-Instruct-v0.2` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| qwen2 / qwen2.5             | `Qwen/Qwen2-7B-Instruct` | `Qwen/Qwen2.5-0.5B-Instruct` |
| gemma2 9B                   | `google/gemma-2-9b-it` | `google/gemma-2-2b-it` (gated) |

Train all: `anima train-zoo --tier all` (GPU required for `large` tier).

You still need HF access, enough RAM/VRAM, and **your own** trained `probes/zoo/<slug>_*.pt` for meaningful readouts.

## Two different things: `layer_config` vs `probes/zoo`

| | `core/layer_config.py` | `probes/zoo/*.pt` |
|--|------------------------|-------------------|
| **What it is** | Which **layers** to hook and `hidden_dim` per HF model id | **Trained** probe weights (valence / arousal / uncertainty heads) |
| **Shipped in repo?** | Yes (metadata only) | **No** for Llama/Mistral/Qwen/Gemma — only `.meta.json` placeholders + local tiny checkpoint |
| **If missing** | API returns `unknown_model` | Probe is **random** → weak / neutral readouts |

### Probe checkpoints that exist today

| HF model | Zoo file (local / release) | `probe_origin` |
|----------|----------------------------|----------------|
| `hf-internal-testing/tiny-random-gpt2` | `tiny_random_gpt2.pt` (run `scripts/build_zoo_tiny_probe.py`) | `synthetic_tiny` |
| `distilgpt2` | `distilgpt2_text.pt` — **train**: `anima train-text` | `text_emotion` |
| `distilgpt2` | `distilgpt2_narratives_pca.pt` — **train**: `anima train` + Narratives | `narratives_fMRI` |
| `meta-llama/Llama-3.2-1B-Instruct` (CPU proxy) | `*_text.pt` via `anima train-text-all` | `text_emotion` |
| `Qwen/Qwen2.5-0.5B-Instruct`, `google/gemma-2-2b-it` | same | `text_emotion` |
| Llama-3-8B / Mistral-7B / Qwen2-7B / Gemma-9B | **GPU only** — `ANIMA_TRAIN_LARGE=1` | train on GPU machine |

Slug rule: last segment of the HF id, lowercased, hyphens → underscores (e.g. `Meta-Llama-3-8B-Instruct` → `meta_llama_3_8b_instruct`).

## How to get real weights for a large HF model

```powershell
# Text-emotion (GoEmotions, no fMRI)
anima train-text --model meta-llama/Meta-Llama-3-8B-Instruct --max-samples 500

# Brain-aligned (needs NARRATIVES_ROOT)
$env:NARRATIVES_ROOT="C:\data\ds002345"
anima train --narratives-root $env:NARRATIVES_ROOT --model meta-llama/Meta-Llama-3-8B-Instruct
```

Checkpoints are gitignored (`*.pt`); publish via GitHub Release and `scripts/download_zoo.py` when ready.
