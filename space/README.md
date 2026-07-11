---
title: Anima Readout Demo
emoji: 🧠
colorFrom: blue
colorTo: cyan
sdk: gradio
app_file: app.py
pinned: false
---

# Anima HF Space (v2.1)

Public demo for **dimensional readout** from Hugging Face causal LMs — not claims that models "feel" emotions.

**Live:** [huggingface.co/spaces/sidb078/Anima](https://huggingface.co/spaces/sidb078/Anima)  
**Source repo:** [github.com/Siddarthb07/Anima](https://github.com/Siddarthb07/Anima)

## Deploy notes

1. **Hero model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (best prompt separation). Fallback: `hf-internal-testing/tiny-random-gpt2` for low RAM.
2. **Probe weights:** `python scripts/download_zoo.py --skip-existing` at build time.
3. **Public mode (required on Space):** `ANIMA_PUBLIC_MODE=1`
4. **Stack:** Standalone **Gradio** (`app.py`) — not the full FastAPI+React Docker stack on free CPU tier.
5. **Limits:** [USAGE_AND_LIMITATIONS.md](https://github.com/Siddarthb07/Anima/blob/main/docs/USAGE_AND_LIMITATIONS.md)

## Local equivalent

```bash
pip install -e ".[gradio]"
python scripts/download_zoo.py --skip-existing
ANIMA_PUBLIC_MODE=1 python space/app.py
```

## v2 features to demo

- Rolling **stability score** + `guard_mode: gate`
- Opt-in **`intervention_mode: dampen`**
- POC prompts: `benchmarks/fixtures/poc_emotional_prompts.json`
