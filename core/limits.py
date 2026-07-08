"""Request bounds and public-demo model policy for the API."""

from __future__ import annotations

import os

# Hard caps (override via env for local research; keep defaults safe for public Spaces).
MAX_NEW_TOKENS = int(os.environ.get("ANIMA_MAX_NEW_TOKENS", "512"))
MAX_PROMPT_CHARS = int(os.environ.get("ANIMA_MAX_PROMPT_CHARS", "8000"))
MAX_ENCODE_LENGTH = int(os.environ.get("ANIMA_MAX_ENCODE_LENGTH", "512"))

# CPU-safe models allowed when ANIMA_PUBLIC_MODE=1 (HF Space / portfolio demo).
PUBLIC_DEMO_MODELS: frozenset[str] = frozenset(
    {
        "hf-internal-testing/tiny-random-gpt2",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
)


def public_mode_enabled() -> bool:
    return os.environ.get("ANIMA_PUBLIC_MODE", "").strip().lower() in ("1", "true", "yes")


def clamp_max_new_tokens(n: int) -> int:
    return max(1, min(int(n), MAX_NEW_TOKENS))


def validate_prompt(prompt: str) -> None:
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt exceeds {MAX_PROMPT_CHARS} characters")


def clamp_encode_length(length: int | None) -> int | None:
    if length is None:
        return None
    return max(1, min(int(length), MAX_ENCODE_LENGTH))


def assert_model_allowed(model_name: str) -> None:
    if public_mode_enabled() and model_name not in PUBLIC_DEMO_MODELS:
        raise ValueError(
            f"model not enabled on public demo ({model_name}); "
            f"allowed: {', '.join(sorted(PUBLIC_DEMO_MODELS))}"
        )
