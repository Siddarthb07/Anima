# Getting started with Anima

Clone from GitHub and run the bootstrap script (works on Linux, macOS, and Windows):

```bash
git clone https://github.com/Siddarthb07/Anima.git
cd Anima
python scripts/bootstrap.py
# or: anima bootstrap
```

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

### Dashboard not loading?

1. **API must run first** on the same port as `dashboard/.env` (`VITE_API_HTTP_TARGET=http://127.0.0.1:8010`).
2. **Docker optional** — if Docker Desktop is down, use native `anima api` + `npm run dev` (not `docker compose`).
3. Copy `dashboard/.env.example` → `dashboard/.env` if missing.
4. Check **http://127.0.0.1:8010/health** then open the Vite URL.
5. Screenshot guide: [`images/dashboard-websocket-troubleshooting.png`](images/dashboard-websocket-troubleshooting.png).

`GET /models` lists each HF id with `brain_data_tier` (`synthetic_minimal` vs `real_fMRI`).

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

## Docker Compose (v2 stack)

From the repo root — download probes first, then start the **stack** profile:

```bash
python scripts/download_zoo.py
docker compose --profile pull run --rm model-pull   # optional: cache HF weights
```

**Windows:** `.\scripts\docker-up.ps1 qwen` → dashboard **http://localhost:8080**, API **http://localhost:8010**

**Linux / macOS:** `chmod +x scripts/docker-up.sh && ./scripts/docker-up.sh qwen`

Stop: `.\scripts\docker-down.ps1` or `./scripts/docker-down.sh`

The dashboard container nginx-proxies `/ws`, `/models`, and `/generate` to the API service.

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
