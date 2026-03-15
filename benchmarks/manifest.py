from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def reports_dir() -> Path:
    return Path(__file__).resolve().parent / "reports"


def write_manifest(model: str, entries: list[dict[str, Any]], *, git_sha: str = "unknown") -> Path:
    out_dir = reports_dir() / f"{model.replace('/', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": model,
        "git_sha": git_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    latest = reports_dir() / "latest_manifest.json"
    latest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
