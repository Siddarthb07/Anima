"""
Download a laptop-sized OpenNeuro ds002345 (Narratives) subset for Anima brain probes.

Stories required by benchmarks/splits/narratives_holdout.json: pieman, tunnel (train), lucy (holdout).

Uses openneuro-py (Python 3.9-compatible 2023.x). Prefers a few known multi-story
subjects so we do not pull the full ~100GB dump.

Usage:
  python scripts/download_ds002345_subset.py
  python scripts/download_ds002345_subset.py --target data/ds002345 --subjects 001 003 004
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = ROOT / "data" / "ds002345"
REQUIRED_STORIES = ("pieman", "tunnel", "lucy")

# Subjects known to have multiple Narratives stories in ds002345 (pieman cohort overlaps).
# Start small; add more if validation fails.
DEFAULT_SUBJECTS = ("001", "003", "004", "005", "006")


def main() -> int:
    p = argparse.ArgumentParser(description="Download ds002345 subset (pieman/tunnel/lucy)")
    p.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    p.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    p.add_argument("--tag", default=None, help="OpenNeuro snapshot tag (optional)")
    args = p.parse_args()

    try:
        import openneuro
    except ImportError:
        print("Install openneuro-py first: python -m pip install 'openneuro-py==2023.1.0'", file=sys.stderr)
        return 1

    target: Path = args.target
    target.mkdir(parents=True, exist_ok=True)

    include: list[str] = [
        "dataset_description.json",
        "participants.tsv",
        "README",
        "stimuli/*",
    ]
    for sub in args.subjects:
        include.append(f"sub-{sub}/anat/*")
        for story in REQUIRED_STORIES:
            # Preprocessed BOLD + accompanying sidecars under func/
            include.append(f"sub-{sub}/func/*task-{story}*")

    print(f"Downloading ds002345 subset -> {target}")
    print(f"Subjects: {args.subjects}")
    print(f"Stories: {REQUIRED_STORIES}")
    print(f"Include patterns ({len(include)}):")
    for pat in include:
        print(f"  {pat}")

    openneuro.download(
        dataset="ds002345",
        tag=args.tag,
        target_dir=str(target),
        include=include,
        max_concurrent_downloads=4,
    )

    # Mark as real (no synthetic_brain_minimal)
    meta = {
        "source": "openneuro_ds002345",
        "dataset": "ds002345",
        "stories": list(REQUIRED_STORIES),
        "subjects": list(args.subjects),
        "note": "Laptop subset for Anima brain-probe retrain; not the full dump",
    }
    (target / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Validate via subprocess (avoid scripts-as-package import issues)
    r = __import__("subprocess").run(
        [sys.executable, str(ROOT / "scripts" / "validate_narratives_root.py"), "--root", str(target)],
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        print("Validation incomplete — may need more subjects or different snapshot files.", file=sys.stderr)
        return 2

    print(f"\nOK. Set NARRATIVES_ROOT={target.resolve()}")
    env_path = ROOT / ".env.benchmark"
    line = f"NARRATIVES_ROOT={target.as_posix()}"
    if env_path.exists():
        text = env_path.read_text(encoding="utf-8")
        if "NARRATIVES_ROOT=" in text:
            lines = [ln if not ln.startswith("NARRATIVES_ROOT=") else line for ln in text.splitlines()]
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            env_path.write_text(text.rstrip() + "\n" + line + "\n", encoding="utf-8")
        print(f"Updated {env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
