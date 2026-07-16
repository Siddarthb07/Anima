---
title: Anima Readout Demo
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# Anima — LLM affect readouts

Live **valence / arousal / uncertainty** readouts from Hugging Face causal LMs via hidden-state probes.

**Not** claims that models feel emotions — [limitations](https://github.com/Siddarthb07/Anima/blob/main/docs/USAGE_AND_LIMITATIONS.md).

- **Repo:** [github.com/Siddarthb07/Anima](https://github.com/Siddarthb07/Anima)
- **Hero model:** TinyLlama (switch in dropdown; tiny-random-gpt2 default on free CPU for reliability)

## Deploy from main repo

```bash
pip install huggingface_hub
# HF_TOKEN from https://huggingface.co/settings/tokens (write)
python scripts/deploy_hf_space.py
```

Or set GitHub secret `HF_TOKEN` — pushes to `space/**` on `main` run `.github/workflows/hf-space-deploy.yml`.
