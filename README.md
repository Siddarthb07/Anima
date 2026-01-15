# Anima (beta)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Anima** is a small stack I built to stream **dimensional readouts** (things like valence, arousal, and uncertainty-style probe outputs) from Hugging Face causal language models, using hooks on hidden states. It ships with a **FastAPI** backend and a **React + Vite** dashboard so you can see tokens and plots update as the model generates.

This is **beta software**: behavior, defaults, and docs will tighten over time. It’s MIT-licensed so you can fork it, break it, and improve it — just don’t treat the UI as ground truth for whether a model “feels” anything (it doesn’t).

**Quick try**

```powershell
pip install -e ".[dev]"
cd dashboard && npm install && copy .env.example .env
```

Terminal A: `python -m uvicorn api.server:app --host 127.0.0.1 --port 8010`  
Terminal B (from `dashboard/`): `npm run dev` → open **http://127.0.0.1:5173**

On Windows you can also run `powershell -ExecutionPolicy Bypass -File scripts\start_anima.ps1` after the installs above.

---

### Documentation

| Doc | What’s in it |
|-----|----------------|
| [**Getting started**](docs/GETTING_STARTED.md) | Install, ports, default models, REST/WebSocket, Docker, troubleshooting |
| [**Project overview**](docs/PROJECT_OVERVIEW.md) | How the repo is laid out, probes vs random weights, TRIBEv2 surrogate (what it is / isn’t) |
| [**Usage & limitations**](docs/USAGE_AND_LIMITATIONS.md) | What you should and shouldn’t use this for — read before demos or papers |
| [**Commands & tests**](docs/RUN_AND_TEST_COMMANDS.txt) | Copy-paste commands for pytest, CLI, Playwright demo |

Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) · Security: [`SECURITY.md`](SECURITY.md)

---

### License

Released under the [MIT License](LICENSE). Model weights you pull from Hugging Face stay under **their** licenses — Anima only wraps inference you configure.
