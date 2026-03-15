"""Fast internal Narratives dev check (1 story) — not for README headline numbers."""

from __future__ import annotations

import os
from typing import Any


def run(model: str) -> dict[str, Any]:
    root = os.environ.get("NARRATIVES_ROOT", "")
    if not root:
        return {
            "benchmark": "narratives_dev",
            "tier": "internal",
            "status": "skipped",
            "reason": "NARRATIVES_ROOT not set",
            "model": model,
        }
    return {
        "benchmark": "narratives_dev",
        "tier": "internal",
        "status": "skipped",
        "reason": "Use run_narratives_encoding.py for holdout metrics; dev runner reserved for CI subset",
        "model": model,
        "narratives_root": root,
    }
