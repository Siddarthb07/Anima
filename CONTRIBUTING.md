# Contributing to Anima

Thanks for improving **Anima**. This project ships as **beta**; small API/UI tweaks are still fair game — just describe them in PRs so downstream users aren’t surprised.

## Principles

- **Framing:** Don’t imply models literally “feel” emotions. Prefer **readouts**, internal geometry, and clearly flagged psychology analogies (see root `README.md` and `docs/USAGE_AND_LIMITATIONS.md`).
- **Scope:** Keep patches focused; avoid unrelated cleanups in the same PR.
- **Tests:** Run `python -m pytest -q`. Heavy Hugging Face tests are optional: `RUN_HF_TESTS=1`.

## Local setup

```powershell
pip install -e ".[dev]"
cd dashboard
npm install
```

Copy `dashboard/.env.example` → `dashboard/.env`.

## Pull requests

1. What changed and **why**.
2. Any visible behavior change for `/generate`, WebSocket messages, or the dashboard.
3. New dependencies need a short justification.

## Issues

Include OS, Python version, CPU vs CUDA Torch if relevant, model id, and reproduction steps.
