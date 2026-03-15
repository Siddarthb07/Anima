"""Run benchmark tiers and write manifest.json."""

from __future__ import annotations

import argparse
import subprocess

from benchmarks.manifest import write_manifest
from benchmarks.run_go_emotions import run as run_go
from benchmarks.run_halueval_guard import run as run_halu
from benchmarks.run_litcoder_compare import run as run_litcoder
from benchmarks.run_narratives_dev import run as run_narr_dev
from benchmarks.run_narratives_encoding import run as run_narr
from benchmarks.run_smoke import run as run_smoke
from benchmarks.run_truthfulqa_guard import run as run_tqa
from benchmarks.run_tribe_reference import run as run_tribe_ref
from core.defaults import DEFAULT_CAUSAL_LM


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_CAUSAL_LM)
    p.add_argument("--tiers", default="internal,external", help="comma list: internal,external,external_text,external_guard")
    args = p.parse_args()
    tiers = {t.strip() for t in args.tiers.split(",")}
    entries = []

    if "internal" in tiers:
        entries.append(run_smoke(args.model))
        entries.append(run_narr_dev(args.model))
    if "external" in tiers:
        entries.append(run_narr(args.model))
        entries.append(run_litcoder(args.model))
        entries.append(run_tribe_ref(args.model))
        try:
            from benchmarks.run_brainscore import run as run_bs

            entries.append(run_bs(args.model))
        except Exception as exc:
            entries.append({"benchmark": "brainscore_language", "status": "error", "message": str(exc)})
    if "external_text" in tiers:
        entries.append(run_go(args.model))
    if "external_guard" in tiers:
        entries.append(run_halu(args.model))
        entries.append(run_tqa(args.model))

    path = write_manifest(args.model, entries, git_sha=_git_sha())
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
