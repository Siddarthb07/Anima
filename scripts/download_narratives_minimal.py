"""
Try to fetch a small real Narratives (ds002345) slice; fall back to synthetic build.

Real data (recommended for papers):
  pip install openneuro
  openneuro download --dataset ds002345 --target data/ds002345
  set NARRATIVES_ROOT=data/ds002345

This script:
  1) Builds data/narratives_minimal/ (synthetic) if missing — always works offline.
  2) Optionally attempts openneuro download when --fetch-real is passed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYNTH = ROOT / "data" / "narratives_minimal"
REAL = ROOT / "data" / "ds002345"


def _build_synthetic() -> Path:
    import subprocess

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_synthetic_brain_dataset.py")],
        check=True,
        cwd=str(ROOT),
    )
    return SYNTH


def _try_openneuro(target: Path) -> bool:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "openneuro", "-q"],
            check=False,
            capture_output=True,
        )
        target.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [
                "openneuro",
                "download",
                "--dataset",
                "ds002345",
                "--target",
                str(target),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and (target / "stimuli").exists():
            print(f"Downloaded ds002345 to {target}")
            return True
        print("openneuro download did not complete; using synthetic minimal set.")
        if r.stderr:
            print(r.stderr[:500])
    except Exception as exc:
        print(f"openneuro fetch skipped: {exc}")
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fetch-real", action="store_true", help="Attempt full/partial ds002345 download")
    p.add_argument("--root", type=Path, default=None, help="Print recommended NARRATIVES_ROOT")
    args = p.parse_args()

    root = REAL if args.fetch_real and _try_openneuro(REAL) else None
    if root is None:
        root = _build_synthetic()

    print(f"\nSet environment variable:")
    print(f'  NARRATIVES_ROOT={root.as_posix()}')
    env_path = ROOT / ".env.benchmark"
    if env_path.exists():
        text = env_path.read_text(encoding="utf-8")
        line = f"NARRATIVES_ROOT={root.as_posix()}"
        if "NARRATIVES_ROOT=" in text:
            lines = [ln if not ln.startswith("NARRATIVES_ROOT=") else line for ln in text.splitlines()]
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            env_path.write_text(text.rstrip() + "\n" + line + "\n", encoding="utf-8")
        print(f"Updated {env_path}")


if __name__ == "__main__":
    main()
