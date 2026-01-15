"""Shared defaults for local runs.

`DEFAULT_CAUSAL_LM` is tiny (safetensors-friendly), loads on constrained RAM / Windows paging limits,
and keeps pipeline behaviour deterministic enough for dashboard QA.

Use `REFERENCE_CAUSAL_LM_DISTIL` only when the machine can load ~350MB FP16 weights and HF cache works.
"""

DEFAULT_CAUSAL_LM = "hf-internal-testing/tiny-random-gpt2"
REFERENCE_CAUSAL_LM_DISTIL = "distilgpt2"
