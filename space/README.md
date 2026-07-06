---
title: Anima Readout Demo
emoji: 🧠
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
---

# Anima HF Space (v1.2)

Public demo for **dimensional readout** from Hugging Face causal LMs — not claims that models "feel" emotions.

## Deploy notes

1. **Default model:** `hf-internal-testing/tiny-random-gpt2` (low RAM on free CPU). For coherent English, use `Qwen/Qwen2.5-0.5B-Instruct` if RAM allows (~3 GB).
2. **Probe weights:** run `python scripts/download_zoo.py --skip-existing` at build time, or train locally (`anima train-text`).
3. **Stack:** FastAPI (`anima api`) + optional Gradio proxy (`scripts/gradio_demo.py`) or ship the Vite dashboard as static files behind the API.
4. **Limits:** See [USAGE_AND_LIMITATIONS.md](https://github.com/Siddarthb07/Anima/blob/main/docs/USAGE_AND_LIMITATIONS.md).

## Local equivalent

```bash
pip install -e .
python scripts/download_zoo.py --skip-existing
anima api --port 8010
cd dashboard && npm install && npm run dev
```

## v1.2 features

- Rolling **stability score** + `guard_mode: gate`
- Opt-in **`intervention_mode: dampen`** steering
- POC prompts: `benchmarks/fixtures/poc_emotional_prompts.json`
