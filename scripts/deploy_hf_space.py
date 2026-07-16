#!/usr/bin/env python3
"""Deploy space/ to the Hugging Face Space (sidb078/Anima).

Docker Space: uploads Dockerfile, entrypoint.sh, README.md and removes the old
Gradio app.py / requirements.txt so the original dashboard is what runs.

Usage:
  set HF_TOKEN=hf_...
  python scripts/deploy_hf_space.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_DIR = REPO_ROOT / "space"
SPACE_ID = os.environ.get("ANIMA_HF_SPACE", "sidb078/Anima")
FILES = ("Dockerfile", "entrypoint.sh", "README.md")
REMOVE = ("app.py", "requirements.txt")


def main() -> int:
    missing = [f for f in FILES if not (SPACE_DIR / f).exists()]
    if missing:
        print(f"Missing in space/: {', '.join(missing)}", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    print(f"Uploading Docker Space files to {SPACE_ID} ...")
    for name in FILES:
        path = SPACE_DIR / name
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=SPACE_ID,
            repo_type="space",
            commit_message=f"deploy: sync {name} (original Anima dashboard)",
        )
        print(f"  ok {name}")

    for name in REMOVE:
        try:
            api.delete_file(
                path_in_repo=name,
                repo_id=SPACE_ID,
                repo_type="space",
                commit_message=f"deploy: remove legacy Gradio {name}",
            )
            print(f"  deleted {name}")
        except Exception as exc:
            print(f"  skip delete {name}: {exc}")

    print(f"Done. Space: https://huggingface.co/spaces/{SPACE_ID}")
    print("Note: set Space hardware to CPU (not ZeroGPU) in Settings if build fails.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
