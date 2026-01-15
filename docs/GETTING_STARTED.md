# Getting started with Anima

This guide assumes you’ve cloned the **Anima** repo (GitHub project name can be `Anima`; your local folder name may differ).

## What you need

- Python **3.9+**
- **PyTorch** and **transformers** (pulled in via `pip install -e ".[dev]"` from `pyproject.toml`)
- **Node.js** if you want the dashboard (`dashboard/`)

First-time runs download Hugging Face weights into your cache — you’ll need network access unless everything is already cached.

---

## Install

From the repository root:

```powershell
pip install -e ".[dev]"
```

Library-only experiments can skip `[dev]` (you lose pytest extras listed in `pyproject.toml`).

### Dashboard dependencies

```powershell
cd dashboard
npm install
copy .env.example .env
```

Edit `dashboard/.env` if your API isn’t on `http://127.0.0.1:8010` (`VITE_API_HTTP_TARGET` must match uvicorn).

---

## Run the API

Default local port used in docs and `.env.example` is **8010**:

```powershell
cd <repo-root>
python -m uvicorn api.server:app --host 127.0.0.1 --port 8010
```

Check **http://127.0.0.1:8010/health** — you should see `{"status":"ok"}`. Interactive docs: **http://127.0.0.1:8010/docs**.

---

## Run the dashboard (development)

```powershell
cd dashboard
npm run dev
```

Open **http://127.0.0.1:5173** (or the URL Vite prints).

In dev, the browser talks to **`ws://127.0.0.1:5173/ws/...`** and Vite **proxies** `/ws` to `VITE_API_HTTP_TARGET`. After changing `.env`, restart `npm run dev`.

Optional direct WebSocket (skip proxy): set `VITE_WS_DIRECT=1` and `VITE_WS_BASE` — see `.env.example`.

---

## Default model (reliable on modest RAM)

The API default is **`hf-internal-testing/tiny-random-gpt2`** (`core/defaults.py`). It’s a tiny regression checkpoint: **decoded text looks like noise**, but hooks, streaming, and the dashboard pipeline run without needing a big GPU.

For coherent English, switch the dashboard field (or JSON `model`) to **`distilgpt2`** or another id listed in `core/layer_config.py` once your machine can load the weights.

---

## One-shot Windows helper

From repo root (after Python + dashboard `npm install`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_anima.ps1
```

That opens two windows: uvicorn on **8010** and Vite on **5173**.

---

## REST smoke test

With the API running:

```powershell
curl -X POST http://127.0.0.1:8010/generate -H "Content-Type: application/json" -d "{\"prompt\":\"Hello\",\"max_new_tokens\":8,\"detect_suppression\":true}"
```

Omitting `model` uses the default tiny checkpoint.

---

## Docker Compose

From the repo root:

```powershell
docker compose build
docker compose up
```

Compose typically exposes the API on **8000** and serves the built dashboard on **8080** — see `docker-compose.yml`. Rebuild the dashboard image with the correct **`VITE_WS_BASE`** if you’re not opening the UI from localhost.

---

## Tests

```powershell
python -m pytest -q
```

Integration tests that download Hub models:

```powershell
$env:RUN_HF_TESTS="1"; python -m pytest -q
```

More detail: [`RUN_AND_TEST_COMMANDS.txt`](RUN_AND_TEST_COMMANDS.txt).

---

## When something breaks

| Symptom | Things to check |
|--------|-------------------|
| WebSocket errors in the UI | API running? Port matches `VITE_API_HTTP_TARGET`? Restart Vite after `.env` edits. |
| Windows crash / access violation loading Torch | Smaller model, updated CPU wheel, paging file / RAM. |
| Everything reads “neutral” in the UI | With **no** `probes/zoo/<slug>.pt`, the probe is **random** — train one or expect weak quadrant labels. See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md). |

For ethics and product boundaries, see [USAGE_AND_LIMITATIONS.md](USAGE_AND_LIMITATIONS.md).
