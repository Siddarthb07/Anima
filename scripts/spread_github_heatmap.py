#!/usr/bin/env python3
"""
Spread commit author/committer dates across a calendar range for GitHub heatmap.

Rewrites history (force-push required). Skips forks and repos in SKIP_REPOS.

  python scripts/spread_github_heatmap.py --dry-run
  python scripts/spread_github_heatmap.py --push
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

OWNER = "Siddarthb07"
SKIP_REPOS: set[str] = set()
# Archived on GitHub are read-only; unarchive first if you need to rewrite them.
ARCHIVED_SKIP = {
    "webcam-sketcher",
    "cv2-volume-control",
    "AI-powered-whatsapp-chatbot",
    "AI-Risk-Prediction-",
    "health-tracker-v2",
}
START = datetime(2026, 1, 15, 10, 0, 0)
END = datetime(2026, 5, 22, 18, 0, 0)
WORK = Path(os.environ.get("TEMP", "/tmp")) / "anima-heatmap-spread"


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> str:
    r = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr or r.stdout}")
    return (r.stdout or "").strip()


def list_repos() -> list[str]:
    out = run(["gh", "repo", "list", OWNER, "--limit", "100", "--json", "name,isFork"])
    data = json.loads(out)
    return [r["name"] for r in data if not r.get("isFork") and not r.get("isArchived")]


def clone_repo(name: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
        if dest.exists():
            run(["cmd", "/c", "rmdir", "/s", "/q", str(dest)], check=False)
    run(["git", "clone", "--quiet", f"https://github.com/{OWNER}/{name}.git", str(dest)])
    return dest


def commit_hashes(repo: Path) -> list[str]:
    out = run(["git", "rev-list", "--reverse", "HEAD"], cwd=repo)
    return [h for h in out.splitlines() if h]


def spread_dates(n: int, start: datetime, end: datetime) -> list[datetime]:
    if n <= 0:
        return []
    span = (end - start).days
    if n == 1:
        return [start + timedelta(days=span // 2)]
    step = max(1, span // max(1, n - 1))
    dates = []
    d = start
    for i in range(n):
        dates.append(d)
        d = min(end, d + timedelta(days=step))
    return dates


def rewrite_dates(repo: Path, mapping: dict[str, str]) -> None:
    parts = []
    for h, iso in mapping.items():
        # git filter-branch env-filter (bash); Git for Windows includes bash
        parts.append(
            f'if [ "$GIT_COMMIT" = "{h}" ]; then '
            f'export GIT_AUTHOR_DATE="{iso}"; export GIT_COMMITTER_DATE="{iso}"; fi'
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


def process_repo(name: str, day_cursor: datetime, *, dry_run: bool, push: bool) -> datetime:
    dest = WORK / name
    print(f"\n=== {name} ===", flush=True)
    if dry_run:
        # estimate from API
        try:
            out = run(
                ["gh", "api", f"repos/{OWNER}/{name}/commits?per_page=100", "--paginate", "-q", ".[].sha"],
                check=False,
            )
            n = len([h for h in out.splitlines() if h.strip()])
        except Exception:
            n = 0
        print(f"  would spread {n} commits from {day_cursor.date()}", flush=True)
        return day_cursor + timedelta(days=max(1, n))

    clone_repo(name, dest)
    hashes = commit_hashes(dest)
    if not hashes:
        print("  no commits, skip", flush=True)
        return day_cursor

    span_days = max(1, (END - START).days)
    mapping = {}
    for i, h in enumerate(hashes):
        if len(hashes) <= 1:
            frac = 0.5
        else:
            frac = i / (len(hashes) - 1)
        sec = int((END - START).total_seconds() * frac)
        dt = START + timedelta(seconds=sec)
        # Stagger same-day commits by hours when many land on one calendar day
        dt = dt + timedelta(hours=(i % 8))
        mapping[h] = dt.strftime("%a %b %d %H:%M:%S %Y +0000")

    rewrite_dates(dest, mapping)
    print(f"  rewrote {len(hashes)} commits", flush=True)
    if push:
        run(["git", "push", "--force-with-lease", "origin", "HEAD"], cwd=dest)
        print("  pushed", flush=True)
    shutil.rmtree(dest, ignore_errors=True)
    return d + timedelta(days=1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--push", action="store_true", help="force-push rewritten history")
    p.add_argument("--repos", nargs="*", help="subset of repo names")
    args = p.parse_args()

    if not shutil.which("gh"):
        print("Install GitHub CLI (gh) and authenticate: gh auth login", file=sys.stderr)
        return 1
    if not shutil.which("git"):
        print("git not found", file=sys.stderr)
        return 1

    repos = args.repos or list_repos()
    repos = [r for r in repos if r not in SKIP_REPOS and r not in ARCHIVED_SKIP]
    # priority: recent bulk-push repos first
    priority = [
        "text2sql-rag",
        "NeuralVortex",
        "project_thrive",
        "Lexprobe",
        "Health-AI",
        "GeoQuant",
        "Drone-Vortex-Ring-Simulation",
        "Propeller-simulator",
        "vortex-tracker",
        "Siddarthb07",
        "siddarthb",
        "homelab-rpi",
        "sign-language-cv",
        "quad-build-log",
    ]
    ordered = [r for r in priority if r in repos]
    ordered += [r for r in repos if r not in ordered]

    cursor = START
    for name in ordered:
        try:
            cursor = process_repo(name, cursor, dry_run=args.dry_run, push=args.push)
            if cursor > END:
                cursor = START + timedelta(days=((cursor - START).days % (END - START).days))
        except Exception as exc:
            print(f"  ERROR {name}: {exc}", flush=True)

    print("\nDone.", flush=True)
    if not args.push and not args.dry_run:
        print("Re-run with --push to apply (rewrites history on remote).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
