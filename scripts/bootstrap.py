#!/usr/bin/env python3
"""
First-time setup for any clone of Anima (cross-platform).

  python scripts/bootstrap.py          # install + minimal data + train tiny probes + pytest
  python scripts/bootstrap.py --skip-train   # install + data only
  python scripts/bootstrap.py --skip-tests
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], **kwargs) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT), **kwargs)


def main() -> int:
    p = argparse.ArgumentParser(description="Bootstrap Anima for local development")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--skip-dashboard", action="store_true", help="Skip npm install")
    args = p.parse_args()

    py = sys.executable
    if run([py, "-m", "pip", "install", "-e", ".[dev]"]) != 0:
        return 1

    if not args.skip_dashboard and (ROOT / "dashboard" / "package.json").exists():
        npm = "npm.cmd" if os.name == "nt" else "npm"
        if run([npm, "install"], cwd=str(ROOT / "dashboard")) != 0:
            print("warn: dashboard npm install failed — continue without UI", flush=True)
        env_ex = ROOT / "dashboard" / ".env.example"
        env_out = ROOT / "dashboard" / ".env"
        if env_ex.exists() and not env_out.exists():
            env_out.write_text(env_ex.read_text(encoding="utf-8"), encoding="utf-8")

    if run([py, str(ROOT / "scripts" / "download_narratives_minimal.py")]) != 0:
        return 1

    narr = ROOT / "data" / "narratives_minimal"
    os.environ["NARRATIVES_ROOT"] = str(narr.resolve())

    # Phase 2: prefer published Release weights; fall back to local train
    dl_code = run([py, str(ROOT / "scripts" / "download_zoo.py"), "--skip-existing"])
    has_pt = any(ROOT.glob("probes/zoo/*.pt"))

    if not args.skip_train and (dl_code != 0 or not has_pt):
        code = run([py, str(ROOT / "scripts" / "train_all_probes.py")])
        if code != 0:
            print("warn: full train failed; trying tiny synthetic probe only", flush=True)
            run([py, str(ROOT / "scripts" / "build_zoo_tiny_probe.py")])
            run([py, str(ROOT / "scripts" / "train_text_lite.py"), "--max-samples", "40"])
    elif has_pt:
        print("Using probe weights from probes/zoo/ (release or existing)", flush=True)

    run([py, str(ROOT / "scripts" / "setup_benchmarks.py")])

    if not args.skip_tests:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "ANIMA_FORCE_CPU": "1"}
        if run([py, "-m", "pytest", "-q", "-k", "not distilgpt2"], env=env) != 0:
            return 1

    print("\nBootstrap complete.")
    print("  API:    anima api --port 8010")
    print("  UI:     cd dashboard && npm run dev")
    print("  Models: Hugging Face ids in core/layer_config.py (not Ollama)")
    print("  Docs:   docs/MODELS_AND_ZOO.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
