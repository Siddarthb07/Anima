# Training probes on your machine

## What failed on a typical Windows laptop (no GPU, low page file)

| Issue | Fix |
|-------|-----|
| `paging file is too small` (os error 1455) | Increase virtual memory (System → Performance → Advanced → Virtual memory). Close browsers. Reboot. Then retry `anima train-text --model distilgpt2`. |
| `gated repo` for Llama / Gemma | `pip install huggingface_hub` then `huggingface-cli login` and accept model licenses on HF. |
| 7B / 8B / 9B OOM on CPU | Use a **GPU** machine: `set ANIMA_TRAIN_LARGE=1`, `set ANIMA_LOAD_8BIT=1`, `set ANIMA_FORCE_CPU=0`, then `python scripts/train_text_zoo_all.py --tier large`. |

## Commands

```powershell
# Benchmark env + guard fixtures
python scripts/setup_benchmarks.py

# Low RAM: tiny model + GoEmotions
python scripts/train_text_lite.py --max-samples 60

# Normal CPU tier (open models)
python scripts/train_text_zoo_all.py --tier cpu --max-samples 100

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
