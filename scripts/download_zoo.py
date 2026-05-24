"""Download probe zoo checkpoints from GitHub Releases."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ZOO = Path(__file__).resolve().parent.parent / "probes" / "zoo"
REPO = "Siddarthb07/Anima"
DEFAULT_TAG = "v1.1.0"

# CPU-tier weights published on Release (meta.json stays in git)
RELEASE_ASSETS: dict[str, str] = {}


def _tag() -> str:
    return os.environ.get("ANIMA_ZOO_RELEASE", DEFAULT_TAG)


def _asset_url(name: str, tag: str | None = None) -> str:
    t = tag or _tag()
    return f"https://github.com/{REPO}/releases/download/{t}/{name}"


def _populate_assets(tag: str | None = None) -> dict[str, str]:
    names = [
        "distilgpt2_text.pt",
        "distilgpt2_narratives_pca.pt",
        "distilgpt2_narratives_pca.calib.pt",
        "distilgpt2_tribe_proj.npz",
        "tiny_random_gpt2.pt",
        "tiny_random_gpt2_text.pt",
        "tiny_random_gpt2_narratives_pca.pt",
        "tiny_random_gpt2_narratives_pca.calib.pt",
        "tiny_random_gpt2_tribe_proj.npz",
    ]
    return {n: _asset_url(n, tag) for n in names}


def download_one(name: str, url: str, dest: Path) -> None:
    print(f"Downloading {name}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest}")


def main() -> int:
    p = argparse.ArgumentParser(description="Download published probe zoo from GitHub Releases")
    p.add_argument("--list", action="store_true", help="List release asset URLs")
    p.add_argument("--tag", default=None, help=f"Release tag (default {DEFAULT_TAG})")
    p.add_argument("--skip-existing", action="store_true", help="Skip files already in probes/zoo/")
    args = p.parse_args()

    tag = args.tag or _tag()
    assets = _populate_assets(tag)

    if args.list:
        for name, url in assets.items():
            print(f"{name}\t{url}")
        return 0

    if not assets:
        print("No release assets configured.", file=sys.stderr)
        return 1

    ok, failed = 0, 0
    for name, url in assets.items():
        dest = ZOO / name
        if args.skip_existing and dest.is_file() and dest.stat().st_size > 0:
            print(f"skip existing {name}")
            ok += 1
            continue
        try:
            download_one(name, url, dest)
            ok += 1
        except urllib.error.HTTPError as exc:
            print(f"  failed {name}: HTTP {exc.code}", file=sys.stderr)
            failed += 1
        except Exception as exc:
            print(f"  failed {name}: {exc}", file=sys.stderr)
            failed += 1

    if ok and not failed:
        print(f"Downloaded {ok} files into {ZOO}")
        return 0
    if ok and failed:
        print(f"Partial: {ok} ok, {failed} failed. Train locally or check release {tag}.", file=sys.stderr)
        return 2
    print(f"No assets fetched from {tag}. Train locally:", file=sys.stderr)
    print("  python scripts/bootstrap.py", file=sys.stderr)
    print("  anima train-text --model distilgpt2", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
