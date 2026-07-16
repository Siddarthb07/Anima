#!/usr/bin/env bash
set -euo pipefail

# Probe weights are not in the pip wheel — pull Release assets used by public demo.
python - <<'PY'
import urllib.request
from pathlib import Path

try:
    from probes.zoo_io import ZOO_DIR
except Exception as exc:
    print("probe import failed:", exc)
    raise SystemExit(0)

ZOO_DIR.mkdir(parents=True, exist_ok=True)
base = "https://github.com/Siddarthb07/Anima/releases/download/v2.0.0"
for name in (
    "qwen2.5_0.5b_instruct_text.pt",
    "tinyllama_1.1b_chat_v1.0_text.pt",
    "tiny_random_gpt2_text.pt",
):
    dest = ZOO_DIR / name
    if dest.exists():
        print("have", name)
        continue
    url = f"{base}/{name}"
    print("download", name)
    try:
        urllib.request.urlretrieve(url, dest)
        print("  ok", name)
    except Exception as exc:
        print("  skip", name, exc)
PY

echo "Starting Anima API + dashboard on :7860 (hero=Qwen2.5-0.5B)..."
exec python -m uvicorn api.server:app --host 0.0.0.0 --port 7860
