"""Bundle local probes/zoo/*.pt and *.npz for GitHub Release upload."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZOO = ROOT / "probes" / "zoo"

DEFAULT_NAMES = [
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "dist" / "zoo-v1.1.0-cpu.zip")
    p.add_argument("--also-loose", action="store_true", help="Copy loose files to dist/ for gh release upload")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    missing = [n for n in DEFAULT_NAMES if not (ZOO / n).is_file()]
    if missing:
        print("Missing:", ", ".join(missing))
        return 1

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in DEFAULT_NAMES:
            zf.write(ZOO / name, arcname=name)
            print(f"  + {name}")
    print(f"Wrote {args.out} ({args.out.stat().st_size // 1024} KB)")

    if args.also_loose:
        loose = args.out.parent / "release-assets"
        loose.mkdir(parents=True, exist_ok=True)
        for name in DEFAULT_NAMES:
            dest = loose / name
            dest.write_bytes((ZOO / name).read_bytes())
        print(f"Loose copies in {loose}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
