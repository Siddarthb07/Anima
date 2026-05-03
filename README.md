# Anima

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Open-source instrumentation** for Hugging Face **causal language models**: per-token hidden-state hooks → valence / arousal / uncertainty readouts, optional brain-alignment training, and a live dashboard.

Anima is **not** a chat product and **does not integrate with Ollama**. Point it at a [supported Hugging Face model id](core/layer_config.py) (e.g. `distilgpt2`, `mistralai/Mistral-7B-Instruct-v0.2`) and load or train matching probe weights under `probes/zoo/`.

---

## Who this is for

- Researchers and developers who want **measurable internal readouts** while a model generates text.
- Anyone reproducing or extending **probe training** (GoEmotions text path, Narratives-style fMRI path).
- Contributors — MIT licensed, standard Python + FastAPI + optional Vite UI.

**Not for:** claiming models “feel” emotions, clinical use, or running weights inside Ollama without the HF stack.

---

## Quick start (any OS)

```bash
git clone https://github.com/Siddarthb07/Anima.git
cd Anima
python scripts/bootstrap.py
```

This installs the package, builds the minimal brain-training dataset, trains default **tiny** probes (low RAM), and runs tests.

**Run:**

```bash
anima api --port 8010
# other terminal:
cd dashboard && cp .env.example .env && npm install && npm run dev
```

Open **http://127.0.0.1:5173** (API health: **http://127.0.0.1:8010/health**).

Windows: `powershell -ExecutionPolicy Bypass -File scripts\start_anima.ps1` after bootstrap.

---

## Models & probes

| Question | Answer |
|----------|--------|
| Does it work with Ollama? | **No.** Use the equivalent **Hugging Face** model id. See [docs/MODELS_AND_ZOO.md](docs/MODELS_AND_ZOO.md). |
| Are Mistral/Llama probes pre-trained? | **Not in git** (`*.pt` is gitignored). Run `anima train-text` / `anima train` or `python scripts/train_all_probes.py`. |
| Default model | `hf-internal-testing/tiny-random-gpt2` — small, works on CPU; text output is intentionally noisy. |
| Bigger models | Need RAM/GPU + `huggingface-cli login` for gated weights. [docs/TRAIN_ON_YOUR_MACHINE.md](docs/TRAIN_ON_YOUR_MACHINE.md) |

---

## What it does

1. Load a causal LM from Hugging Face and register **forward hooks** on selected layers.
2. On each generated token, map activations through **trainable probe heads** (valence, arousal, uncertainty).
3. Expose **REST** + **WebSocket** streaming; optional **React dashboard** for live plots.
4. Optional **Narratives-shaped** brain alignment training (`probes/train.py`) and guard / benchmark tooling.

If `probes/zoo/<model_slug>*.pt` is missing, probes are **random** — fine for plumbing, not for scientific claims until you train.

![Dashboard example](docs/images/dashboard-readout-example.png)

---

## CLI

```bash
anima api --port 8010
anima train-zoo --tier cpu
ANIMA_TRAIN_LARGE=1 anima train-zoo --tier large
anima benchmark --tiers internal,external,external_text,external_guard
```

Ollama → HF: `scripts/ollama_to_hf.json` · Full list: `anima --help` · [docs/TRAINING.md](docs/TRAINING.md)

---

## Documentation

| Doc | Contents |
|-----|----------|
| [Getting started](docs/GETTING_STARTED.md) | Install, Docker, API, dashboard |
| [Models & zoo](docs/MODELS_AND_ZOO.md) | HF vs Ollama, checkpoint naming |
| [Training](docs/TRAINING.md) | Text + brain probes |
| [Benchmarks](docs/BENCHMARKS.md) | Manifests and external suites |
| [Project overview](docs/PROJECT_OVERVIEW.md) | Architecture |
| [Usage & limitations](docs/USAGE_AND_LIMITATIONS.md) | Ethics and scope |
| [Contributing](CONTRIBUTING.md) | PRs, tests, conduct |

---

## Development

```bash
python -m pytest -q -k "not distilgpt2"
RUN_HF_TESTS=1 python -m pytest -q   # optional Hub downloads
```

CI: `.github/workflows/ci.yml`

---

## License

[MIT](LICENSE). Hugging Face **model weights** stay under their own licenses; you are responsible for compliance when you download or redistribute them.
