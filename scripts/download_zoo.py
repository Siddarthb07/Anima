"""Download probe zoo assets from GitHub release (placeholder URLs)."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ZOO = Path(__file__).resolve().parent.parent / "probes" / "zoo"

# Populate with release asset URLs when published
RELEASE_ASSETS: dict[str, str] = {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list or not RELEASE_ASSETS:
        print("No pre-built zoo bundle on this mirror yet.")
        print("Train locally (any clone):")
        print("  python scripts/bootstrap.py")
        print("  anima train-text --model distilgpt2")
        print("  anima train --narratives-root ./data/narratives_minimal --model distilgpt2")
        print("Check GitHub Releases for optional .pt assets when published.")
        return
    ZOO.mkdir(parents=True, exist_ok=True)
    for name, url in RELEASE_ASSETS.items():
        dest = ZOO / name
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest}")


if __name__ == "__main__":
    main()
