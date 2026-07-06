#!/usr/bin/env bash
# Start Anima Docker stack (qwen | distil | tiny). Usage: ./scripts/docker-up.sh qwen
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROFILE="${1:-qwen}"
case "$PROFILE" in
  qwen)   export VITE_DEFAULT_MODEL="Qwen/Qwen2.5-0.5B-Instruct" ;;
  distil) export VITE_DEFAULT_MODEL="distilgpt2" ;;
  tiny)   export VITE_DEFAULT_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0" ;;
  *) echo "Usage: $0 [qwen|distil|tiny]" >&2; exit 1 ;;
esac
export ANIMA_WARMUP_MODEL="$VITE_DEFAULT_MODEL"
export VITE_WS_BASE="ws://localhost:8080"
export VITE_API_HTTP_TARGET="http://localhost:8080"

echo "Starting stack '$PROFILE' -> $VITE_DEFAULT_MODEL"
echo "  dashboard http://localhost:8080  |  API http://localhost:8010"
echo "  (First run: python scripts/download_zoo.py && docker compose --profile pull run --rm model-pull)"

docker compose --profile stack up -d --build
echo "Done. Stop with: ./scripts/docker-down.sh"
