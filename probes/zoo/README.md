# Probe zoo

Checkpoints are `*.pt` (gitignored — train locally or download CI artifacts). Sidecars: `*.meta.json`.

## Train all OSS families (Hugging Face, not Ollama)

```bash
python scripts/download_narratives_minimal.py
anima train-zoo --tier cpu          # open models ≤~2B
ANIMA_TRAIN_LARGE=1 anima train-zoo --tier large   # 7B–9B on GPU
```

Ollama name → HF id: [`scripts/ollama_to_hf.json`](../../scripts/ollama_to_hf.json)

## Model families

| Family | HF id (full) | CPU proxy (open) | Zoo files |
|--------|----------------|------------------|-----------|
| GPT-2 | `distilgpt2` | — | `distilgpt2_text.pt`, `distilgpt2_narratives_pca.pt` |
| Llama | `meta-llama/Meta-Llama-3-8B-Instruct` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `*_text.pt`, `*_narratives_pca.pt` per slug |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.2` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | same |
| Qwen | `Qwen/Qwen2-7B-Instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | same |
| Gemma | `google/gemma-2-9b-it` | gated `google/gemma-2-2b-it` | HF login required |

| Tiny (default) | `hf-internal-testing/tiny-random-gpt2` | `python scripts/train_all_probes.py` |

Slug = last path segment of HF id, lowercased, `-` → `_` (e.g. `mistralai/Mistral-7B-Instruct-v0.2` → `mistral_7b_instruct_v0_2`).

## CI-built weights

GitHub Actions [`.github/workflows/train-zoo.yml`](../.github/workflows/train-zoo.yml) builds additional probes on Ubuntu (GPU for 7B+).

**Published Release (CPU tier):**

```bash
python scripts/download_zoo.py
```

Assets on [v2.0.0](https://github.com/Siddarthb07/Anima/releases/tag/v2.0.0): v1.1.0 CPU set plus **Qwen2.5-0.5B**, **TinyLlama-1.1B**, and **SmolLM2-1.7B** text probes. Older tags: [v1.1.0](https://github.com/Siddarthb07/Anima/releases/tag/v1.1.0) (base CPU), `ANIMA_ZOO_RELEASE=v1.1.1` for Qwen-only delta.

Download CI artifacts from a workflow run and copy into this folder for proxy / 7B builds.
