# Training probes on your machine

## RAM tiers (physical memory)

Windows also needs a large enough **paging file** (virtual memory). With **16 GB RAM**, set paging file to **16–32 GB** (custom) if you see `os error 1455` — that error is often page file, not raw RAM.

| Physical RAM | Typical local ceiling (CPU, one model at a time) |
|--------------|-----------------------------------------------------|
| 8 GB | `tiny-random-gpt2`, maybe `distilgpt2` with care |
| **16 GB** | **`distilgpt2`**, **`Qwen/Qwen2.5-0.5B-Instruct`**, likely **`TinyLlama-1.1B`** text train; **`SmolLM2-1.7B`** try with low `--max-samples` |
| 32 GB+ | Same CPU proxies more comfortably; still use GPU for 7B+ |

**16 GB build order (recommended):**

1. `distilgpt2` — text + brain + live benchmark + API demo  
2. `Qwen/Qwen2.5-0.5B-Instruct` — `anima train-text --model ... --max-samples 200`  
3. `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — same, one model per session  
4. `HuggingFaceTB/SmolLM2-1.7B-Instruct` — last; reduce `--max-samples` to 100 if OOM  

Close Chrome, pause OneDrive on the repo folder, reboot before long trains.

**Still not realistic on 16 GB CPU:** Llama-3-8B, Mistral-7B, Qwen2-7B, Gemma-9B — use GitHub Actions `train-zoo.yml` or a GPU cloud box.

## What failed on a typical Windows laptop (no GPU, low page file)

| Issue | Fix |
|-------|-----|
| `paging file is too small` (os error 1455) | Increase virtual memory (System → Performance → Advanced → Virtual memory). Close browsers. Reboot. Then retry `anima train-text --model distilgpt2`. For **Qwen 0.5B**, start with `--max-samples 200` before 800. |
| Tokenizer OOM on first load | Text-only train: `$env:ANIMA_SLOW_TOKENIZER="1"` before `anima train-text`. **Brain / Narratives train needs fast tokenizer** — run after text train finishes (frees RAM). |
| `gated repo` for Llama / Gemma | `pip install huggingface_hub` then `huggingface-cli login` and accept model licenses on HF. |
| 7B / 8B / 9B OOM on CPU | Use a **GPU** machine: `set ANIMA_TRAIN_LARGE=1`, `set ANIMA_LOAD_8BIT=1`, `set ANIMA_FORCE_CPU=0`, then `python scripts/train_text_zoo_all.py --tier large`. |
| CPU int8 inference (TinyLlama) | `$env:ANIMA_LOAD_DYNAMIC_INT8="1"` + `$env:ANIMA_FORCE_CPU="1"` before `anima api`. Train probe on fp32 first. |

## Commands

```powershell
# Benchmark env + guard fixtures
python scripts/setup_benchmarks.py

# Low RAM: tiny model + GoEmotions
python scripts/train_text_lite.py --max-samples 60

# Normal CPU tier (open models) — OK on 16 GB if paging file is set
python scripts/train_text_zoo_all.py --tier cpu --max-samples 200

# Single proxy (safer on 16 GB — one model per command)
anima train-text --model Qwen/Qwen2.5-0.5B-Instruct --max-samples 200
anima train-text --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-samples 200

# GPU tier (real Llama-3-8B, Mistral-7B, etc.)
$env:ANIMA_TRAIN_LARGE="1"
$env:ANIMA_LOAD_8BIT="1"
$env:ANIMA_FORCE_CPU="0"
huggingface-cli login
python scripts/train_text_zoo_all.py --tier large

# Narratives fMRI (needs dataset)
$env:NARRATIVES_ROOT="C:\data\ds002345"
anima train --narratives-root $env:NARRATIVES_ROOT --model distilgpt2
```

## Open CPU proxies vs full 7B ids

| Family | CPU proxy (open) | Full id (GPU) |
|--------|------------------|---------------|
| Llama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Qwen | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2-7B-Instruct` |
| Mistral | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `mistralai/Mistral-7B-Instruct-v0.2` |
| Gemma | — (gated) | `google/gemma-2-9b-it` |

Ollama is **not** supported — use the Hugging Face ids above in the Anima API/dashboard.
