from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def reports_dir() -> Path:
    return Path(__file__).resolve().parent / "reports"


MANIFEST_SCHEMA_VERSION = 1


def _probe_slug(model: str) -> str:
    return model.split("/")[-1].lower().replace("-", "_")


def write_manifest(model: str, entries: list[dict[str, Any]], *, git_sha: str = "unknown") -> Path:
    out_dir = reports_dir() / f"{model.replace('/', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "model": model,
        "git_sha": git_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    payload = json.dumps(manifest, indent=2)
    path = out_dir / "manifest.json"
    path.write_text(payload, encoding="utf-8")
    reports_dir().mkdir(parents=True, exist_ok=True)
    (reports_dir() / "latest_manifest.json").write_text(payload, encoding="utf-8")
    (reports_dir() / f"latest_{_probe_slug(model)}_manifest.json").write_text(payload, encoding="utf-8")
    return path
