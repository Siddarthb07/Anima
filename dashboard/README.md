# Anima dashboard

React + Vite front-end for the **Anima** API (streaming token readouts).

Full setup lives in the repo [**README.md**](../README.md) and [**docs/GETTING_STARTED.md**](../docs/GETTING_STARTED.md). Minimal sequence:

1. Repo root: `pip install -e ".[dev]"` and start uvicorn (default **8010**).
2. Here: `npm install`, copy `.env.example` → `.env`, match `VITE_API_HTTP_TARGET`.
3. `npm run dev` and open the URL Vite prints.

Usage boundaries: [**docs/USAGE_AND_LIMITATIONS.md**](../docs/USAGE_AND_LIMITATIONS.md).
