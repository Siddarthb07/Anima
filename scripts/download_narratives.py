"""Pointer script for OpenNeuro ds002345 (Narratives) layout expected by alignment/narratives_loader.py."""

from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED = """
Narratives (ds002345) — minimal dev subset
==========================================

1. Install OpenNeuro CLI or download from https://openneuro.org/datasets/ds002345
2. Point NARRATIVES_ROOT at the dataset root containing per-story folders.

Expected layout (see alignment/narratives_loader.py):
  {NARRATIVES_ROOT}/pieman/...
  {NARRATIVES_ROOT}/tunnel/...
  {NARRATIVES_ROOT}/lucy/...

Train:
  set NARRATIVES_ROOT=C:\\path\\to\\ds002345
  anima train --narratives-root %NARRATIVES_ROOT% --model distilgpt2

Full download is large (~100GB+). Use OpenNeuro selective download for story subsets only.
See also: scripts/download_narratives.md
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Narratives dataset setup helper")
    p.add_argument("--root", type=Path, default=None, help="Verify this path exists")
    args = p.parse_args()
    print(EXPECTED)
    if args.root:
        root = args.root.expanduser()
        if root.is_dir():
            stories = [d.name for d in root.iterdir() if d.is_dir()]
            print(f"Found root {root} with subdirs: {', '.join(sorted(stories)[:12])}")
        else:
            print(f"Path not found: {root}")


if __name__ == "__main__":
    main()
