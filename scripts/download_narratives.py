"""Pointer + layout verifier for OpenNeuro ds002345 (Narratives)."""

from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED = """
Narratives (ds002345) — layout expected by alignment/narratives_loader.py
=======================================================================

Download: https://openneuro.org/datasets/ds002345
Docs:     scripts/download_narratives.md
Validate: python scripts/validate_narratives_root.py --root <path>

Expected layout:
  {NARRATIVES_ROOT}/
    stimuli/
      pieman.txt
      pieman_words.json
      tunnel.txt
      tunnel_words.json
      lucy.txt
      lucy_words.json
    sub-XXX/
      func/
        sub-XXX_task-pieman_*_bold.nii.gz
        ...

NOT valid: top-level per-story folders like {root}/pieman/... (older docs were wrong).

Train (after validate layout_ok=true and NOT synthetic):
  set NARRATIVES_ROOT=C:\\path\\to\\ds002345_subset
  anima train --narratives-root %NARRATIVES_ROOT% --model distilgpt2

Disk: laptop subset ~15–40 GB; full OpenNeuro dump ~100GB+.
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Narratives dataset setup helper")
    p.add_argument("--root", type=Path, default=None, help="Validate this path")
    args = p.parse_args()
    print(EXPECTED)
    if args.root:
        import importlib.util

        vpath = Path(__file__).resolve().parent / "validate_narratives_root.py"
        spec = importlib.util.spec_from_file_location("validate_narratives_root", vpath)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        report = mod.validate_root(args.root.expanduser())
        print(f"layout_ok={report['layout_ok']} subjects={report['n_subjects']}")
        print(f"probe_origin={report['recommended_probe_origin']}")
        for issue in report["issues"]:
            print(f"  - {issue}")


if __name__ == "__main__":
    main()
