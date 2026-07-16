"""Shared defaults for local runs.

`DEFAULT_CAUSAL_LM` is tiny (safetensors-friendly), loads on constrained RAM / Windows paging limits,
and keeps pipeline behaviour deterministic enough for dashboard QA.

Use `REFERENCE_CAUSAL_LM_DISTIL` only when the machine can load ~350MB FP16 weights and HF cache works.
"""

DEFAULT_CAUSAL_LM = "hf-internal-testing/tiny-random-gpt2"
REFERENCE_CAUSAL_LM_DISTIL = "distilgpt2"
# Public demo hero (Space default). TinyLlama remains strongest council score.
HERO_DEMO_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
COUNCIL_BEST_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
