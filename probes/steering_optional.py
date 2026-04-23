"""Optional steering helpers (install steering extras separately)."""

from __future__ import annotations


def steering_vectors_available() -> bool:
    try:
        import steering_vectors  # noqa: F401

        return True
    except ImportError:
        return False


def note() -> str:
    if steering_vectors_available():
        return "steering-vectors package detected; use probes.validate for residual steering."
    return "Install steering-vectors or IBM activation-steering for extended causal tests."
