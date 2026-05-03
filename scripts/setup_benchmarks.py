"""Cross-platform benchmark env setup (guard fixtures + dependency checks)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    fixtures = ROOT / "benchmarks" / "fixtures"
    os.environ.setdefault("TRUTHFULQA_PATH", str(fixtures / "truthfulqa_guard_sample.json"))
    os.environ.setdefault("HALUEVAL_PATH", str(fixtures / "halueval_guard_sample.json"))
    os.environ.setdefault("ANIMA_TRIBE_MODE", "surrogate")

    print("TRUTHFULQA_PATH =", os.environ["TRUTHFULQA_PATH"])
    print("HALUEVAL_PATH =", os.environ["HALUEVAL_PATH"])

    narr = os.environ.get("NARRATIVES_ROOT", "")
    if narr and Path(narr).is_dir():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "download_narratives.py"), "--root", narr], check=False)
    else:
        print("NARRATIVES_ROOT not set — narratives benchmarks will skip.")

    try:
        import brainscore_language  # noqa: F401

        print("brainscore_language: ok")
    except ImportError:
        print("brainscore_language: not installed (set SKIP_BRAINSCORE=1 or pip install brainscore-language)")
        os.environ.setdefault("SKIP_BRAINSCORE", "1")

    try:
        import tribev2  # noqa: F401

        print("tribev2: ok")
    except ImportError:
        print("tribev2: not installed (surrogate-only)")

    print("\nSee docs/MODELS_AND_ZOO.md - Anima does not use Ollama; layer_config != probe weights.")
    env_example = ROOT / ".env.benchmark.example"
    env_out = ROOT / ".env.benchmark"
    if env_example.exists() and not env_out.exists():
        env_out.write_text(env_example.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Created {env_out}")


if __name__ == "__main__":
    main()
