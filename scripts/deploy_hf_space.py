#!/usr/bin/env python3
"""Deploy original Anima dashboard Space (Gradio ZeroGPU + FastAPI dashboard)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_DIR = REPO_ROOT / "space"
SPACE_ID = os.environ.get("ANIMA_HF_SPACE", "sidb078/Anima")


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    dist = SPACE_DIR / "dashboard_dist"
    if not dist.is_dir():
        print("Missing space/dashboard_dist — run: cd dashboard && set VITE_SAME_ORIGIN=1 && npm run build && copy dist", file=sys.stderr)
        return 1

    # Root Gradio/FastAPI entry + deps + metadata
    for name in ("app.py", "requirements.txt", "README.md"):
        api.upload_file(
            path_or_fileobj=str(SPACE_DIR / name),
            path_in_repo=name,
            repo_id=SPACE_ID,
            repo_type="space",
            commit_message=f"deploy: {name}",
        )
        print("ok", name)

    # Built React dashboard (original UI)
    api.upload_folder(
        folder_path=str(dist),
        path_in_repo="dashboard_dist",
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="deploy: original Anima dashboard build",
    )
    print("ok dashboard_dist/")

    # Drop Docker-only leftovers if present
    for name in ("Dockerfile", "entrypoint.sh"):
        try:
            api.delete_file(
                path_in_repo=name,
                repo_id=SPACE_ID,
                repo_type="space",
                commit_message=f"deploy: remove {name}",
            )
            print("deleted", name)
        except Exception as exc:
            print("skip delete", name, exc)

    print(f"Done: https://huggingface.co/spaces/{SPACE_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
