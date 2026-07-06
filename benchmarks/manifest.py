from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def reports_dir() -> Path:
    return Path(__file__).resolve().parent / "reports"


MANIFEST_SCHEMA_VERSION = 1


def _probe_slug(model: str) -> str:
    return model.split("/")[-1].lower().replace("-", "_")


def _repo_relative(path: str | Path) -> str:
    """Store fixture paths relative to repo root when possible."""
    p = Path(path)
    if not p.is_absolute():
        return str(p).replace("\\", "/")
    try:
        root = Path(__file__).resolve().parent.parent
        return str(p.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    if "fixture" in out and isinstance(out["fixture"], str):
        out["fixture"] = _repo_relative(out["fixture"])
    return out


def _json_safe(obj: Any) -> Any:
    import math

    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def write_manifest(model: str, entries: list[dict[str, Any]], *, git_sha: str = "unknown") -> Path:
    normalized = [_normalize_entry(e) for e in entries]
    out_dir = reports_dir() / f"{model.replace('/', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _json_safe(
        {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "model": model,
            "git_sha": git_sha,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "entries": normalized,
        }
    )
    if os.environ.get("NARRATIVES_ROOT"):
        for e in normalized:
            if "narratives_root" in e:
                e["narratives_root"] = _repo_relative(os.environ["NARRATIVES_ROOT"])
    payload = json.dumps(manifest, indent=2)
    path = out_dir / "manifest.json"
    path.write_text(payload, encoding="utf-8")
    reports_dir().mkdir(parents=True, exist_ok=True)
    (reports_dir() / "latest_manifest.json").write_text(payload, encoding="utf-8")
    (reports_dir() / f"latest_{_probe_slug(model)}_manifest.json").write_text(payload, encoding="utf-8")
    return path
