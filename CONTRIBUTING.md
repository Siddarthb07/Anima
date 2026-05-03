# Contributing to Anima

Anima is an **open-source** research tool (MIT). Contributions that help others install, train probes, and reproduce benchmarks are welcome.

## Before you start

- Read [docs/USAGE_AND_LIMITATIONS.md](docs/USAGE_AND_LIMITATIONS.md) — do not frame readouts as literal emotions or clinical scores.
- Anima targets **Hugging Face** causal LMs, not Ollama. If you add model support, extend `core/layer_config.py` and document training steps in [docs/MODELS_AND_ZOO.md](docs/MODELS_AND_ZOO.md).

## Local setup

```bash
git clone https://github.com/Siddarthb07/Anima.git
cd Anima
python scripts/bootstrap.py
```

Or manually:

```bash
pip install -e ".[dev]"
python scripts/download_narratives_minimal.py
python scripts/train_all_probes.py   # tiny default model
cd dashboard && npm install && cp .env.example .env
```

## Tests

```bash
python -m pytest -q -k "not distilgpt2"
RUN_HF_TESTS=1 python -m pytest -q   # downloads Hub weights
```

## Pull requests

1. **What** and **why** (one feature or fix per PR when possible).
2. Note API or WebSocket schema changes.
3. New dependencies need justification in the PR body.
4. If you change training or benchmarks, say how to reproduce (`anima train-text`, `anima benchmark`, etc.).

## Adding a new Hugging Face model

1. Add an entry to `core/layer_config.py` (`layers`, `hidden_dim`).
2. Train: `anima train-text --model <hf_id>` and optionally `anima train --narratives-root ...`.
3. Document in `probes/zoo/README.md` — checkpoints are gitignored; describe how to obtain or train them.
4. Do not commit gated or licensed weights without permission.

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
