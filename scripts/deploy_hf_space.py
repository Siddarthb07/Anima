#!/usr/bin/env python3
"""Deploy space/ files to the Hugging Face Space repo (sidb078/Anima).

HF Spaces expect app.py, requirements.txt, and README.md at the **repo root**,
not under space/. This script uploads the three files from space/ via the Hub API.

Usage:
  set HF_TOKEN=hf_...          # Windows
  export HF_TOKEN=hf_...       # Linux/macOS
  python scripts/deploy_hf_space.py

GitHub Actions: add repo secret HF_TOKEN (write token from huggingface.co/settings/tokens)
to enable auto-deploy on push to space/ via .github/workflows/hf-space-deploy.yml.

Or: huggingface-cli login, then run without HF_TOKEN (uses cached token).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_DIR = REPO_ROOT / "space"
SPACE_ID = os.environ.get("ANIMA_HF_SPACE", "sidb078/Anima")
FILES = ("app.py", "requirements.txt", "README.md")


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

    print(f"Uploading {len(FILES)} file(s) to Space {SPACE_ID} ...")
    for name in FILES:
        path = SPACE_DIR / name
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=SPACE_ID,
            repo_type="space",
            commit_message=f"deploy: sync {name} from Anima main",
        )
        print(f"  ok {name}")

    print(f"Done. Space: https://huggingface.co/spaces/{SPACE_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
