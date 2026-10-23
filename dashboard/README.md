# Anima dashboard

React + Vite front-end for the **Anima** API (streaming token readouts).

**v2.2:** Trust Integrity banner (`TrustIntegrityBanner`) reads `GET /trust-badge` / run summary — LOCKED unless a sealed adapted receipt allows unlock. Fixture AUROC alone never paints green.

Full setup lives in the repo [**README.md**](../README.md) and [**docs/GETTING_STARTED.md**](../docs/GETTING_STARTED.md). Minimal sequence:

1. Repo root: `pip install -e ".[dev]"` and start uvicorn (default **8010**). `ANIMA_TRUST_GATE=auto` (default) attaches sealed receipts when present.
2. Here: `npm install`, copy `.env.example` → `.env`, match `VITE_API_HTTP_TARGET`.
3. `npm run dev` and open the URL Vite prints.

Usage boundaries: [**docs/USAGE_AND_LIMITATIONS.md**](../docs/USAGE_AND_LIMITATIONS.md). Paper spine: [**docs/paper/**](../docs/paper/).
