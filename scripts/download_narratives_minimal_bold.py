"""Download a small real-fMRI subset fast: pieman+lucy × 3 subjects (~280MB)."""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "data" / "ds002345"
BUCKET = "fcp-indi"
PREFIX = "data/Projects/narratives/"
STORIES = ("pieman", "lucy")  # skip tunnel (~145MB×3) for depth push
SUBJECTS = ("022", "026", "041")


def get(key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"skip {dest.relative_to(TARGET)} ({dest.stat().st_size})", flush=True)
        return
    url = f"https://{BUCKET}.s3.amazonaws.com/{urllib.parse.quote(key)}"
    print(f"GET {dest.relative_to(TARGET)}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        "1800",
        "-o",
        str(tmp),
        url,
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(dest)
    print(f"  -> {dest.stat().st_size}", flush=True)


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    n = 0
    for sub in SUBJECTS:
        for story in STORIES:
            for suffix in ("_bold.nii.gz", "_bold.json", "_events.tsv"):
                key = f"{PREFIX}sub-{sub}/func/sub-{sub}_task-{story}{suffix}"
                dest = TARGET / f"sub-{sub}" / "func" / f"sub-{sub}_task-{story}{suffix}"
                try:
                    get(key, dest)
                    if suffix.endswith(".nii.gz"):
                        n += 1
                except subprocess.CalledProcessError as exc:
                    print(f"FAIL {key}: {exc}", flush=True)
    meta = {
        "source": "fcp_indi_narratives",
        "dataset": "ds002345",
        "stories": list(STORIES),
        "subjects": list(SUBJECTS),
        "note": "Laptop depth-push subset (pieman+lucy). Tunnel omitted for size/time.",
    }
    (TARGET / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"BOLD count={n}", flush=True)
    # temporary holdout split: train pieman, holdout lucy
    split = ROOT / "benchmarks" / "splits" / "narratives_holdout_depth.json"
    split.write_text(
        json.dumps({"train": ["pieman"], "holdout": ["lucy"], "note": "depth-push subset"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {split}", flush=True)
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_narratives_root.py"), "--root", str(TARGET)],
        cwd=str(ROOT),
    )
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
