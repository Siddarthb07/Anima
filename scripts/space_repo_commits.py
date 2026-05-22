#!/usr/bin/env python3
"""Evenly space all commits on current branch across a date range (filter-branch)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def run(cmd: list[str], cwd: Path, check: bool = True) -> str:
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr or r.stdout}")
    return (r.stdout or "").strip()


def commit_hashes(repo: Path) -> list[str]:
    out = run(["git", "rev-list", "--reverse", "HEAD"], cwd=repo)
    return [h for h in out.splitlines() if h]


def assign_dates(n: int, start: datetime, end: datetime) -> list[datetime]:
    if n <= 0:
        return []
    total_seconds = (end - start).total_seconds()
    if n == 1:
        return [start + timedelta(seconds=total_seconds / 2)]
    out = []
    for i in range(n):
        frac = i / (n - 1)
        sec = int(total_seconds * frac)
        out.append(start + timedelta(seconds=sec))
    return out


def rewrite(repo: Path, mapping: dict[str, str]) -> None:
    parts = []
    for h, when in mapping.items():
        parts.append(
            f'if [ "$GIT_COMMIT" = "{h}" ]; then export GIT_AUTHOR_DATE="{when}"; export GIT_COMMITTER_DATE="{when}"; fi'
        )
    env_filter = " ; ".join(parts) if parts else "true"
    env = {**os.environ, "FILTER_BRANCH_SQUELCH_WARNING": "1"}
    subprocess.run(
        ["git", "filter-branch", "-f", "--env-filter", env_filter, "HEAD"],
        cwd=str(repo),
        check=True,
        env=env,
    )
    orig = repo / ".git" / "refs" / "original"
    if orig.exists():
        shutil.rmtree(orig, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--start", default="2026-01-20T10:00:00")
    p.add_argument("--end", default="2026-05-22T16:00:00")
    p.add_argument("--push", action="store_true")
    args = p.parse_args()
    repo = args.repo.resolve()
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    hashes = commit_hashes(repo)
    dates = assign_dates(len(hashes), start, end)
    mapping = {}
    for h, dt in zip(hashes, dates):
        mapping[h] = dt.strftime("%a %b %d %H:%M:%S %Y +0000")

    print(f"Spacing {len(hashes)} commits from {start.date()} to {end.date()}")
    rewrite(repo, mapping)
    if args.push:
        run(["git", "push", "--force-with-lease", "origin", "HEAD"], cwd=repo)
        print("Pushed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
