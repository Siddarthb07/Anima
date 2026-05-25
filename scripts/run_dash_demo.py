"""
Drive the dashboard through one streamed generation for QA.

Requires:
  - API: uvicorn (default port from ANIMA_API_PORT or LLM_AFFECT_API_PORT, 8010)
  - Dashboard: Vite dev (port from ANIMA_DASH_PORT or LLM_AFFECT_DASH_PORT, default 5173)
  - pip install playwright && python -m playwright install chromium

Env:
  ANIMA_* (preferred) or LLM_AFFECT_* (legacy): API_PORT, DASH_PORT for URLs.
  ANIMA_DEMO_MODEL / LLM_AFFECT_DEMO_MODEL - HF id (defaults to core.defaults.DEFAULT_CAUSAL_LM).
  ANIMA_DEMO_HEADLESS_ONLY / LLM_AFFECT_DEMO_HEADLESS_ONLY=1 - skip headed Chromium.

Runs Chromium (headed first unless ANIMA_DEMO_HEADLESS_ONLY / LLM_AFFECT_DEMO_HEADLESS_ONLY=1). Retries headless and saves a screenshot.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.defaults import DEFAULT_CAUSAL_LM


def _env_int(anim_key: str, legacy_key: str, default: str) -> int:
    raw = os.environ.get(anim_key) or os.environ.get(legacy_key)
    return int(raw.strip() if raw else default)


def _env_str(anim_key: str, legacy_key: str, default: str) -> str:
    return os.environ.get(anim_key) or os.environ.get(legacy_key) or default


API_PORT = _env_int("ANIMA_API_PORT", "LLM_AFFECT_API_PORT", "8010")
DASH_PORT = _env_int("ANIMA_DASH_PORT", "LLM_AFFECT_DASH_PORT", "5173")
DEMO_MODEL = _env_str(
    "ANIMA_DEMO_MODEL",
    "LLM_AFFECT_DEMO_MODEL",
    DEFAULT_CAUSAL_LM,
)
DEMO_PROMPT = (
    "DEMO SEQUENCE: Answer in exactly six short words: what is 2+2? "
    "Then add one encouraging phrase."
)


def wait_http(url: str, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def run(headless: bool) -> Path | None:
    from playwright.sync_api import sync_playwright

    shot = ROOT / "docs" / "dash_demo_last_run.png"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(f"http://127.0.0.1:{DASH_PORT}/", wait_until="domcontentloaded", timeout=120000)
        page.get_by_test_id("model-input").fill(DEMO_MODEL)
        page.get_by_test_id("prompt-input").fill(DEMO_PROMPT)
        page.get_by_test_id("max-tokens").fill("32")
        page.get_by_test_id("stream-btn").click()
        page.wait_for_function(
            """() => document.querySelectorAll('[data-testid="token-stream-panel"] button[type="button"]').length >= 3""",
            timeout=180000,
        )
        time.sleep(2)
        page.screenshot(path=str(shot), full_page=True)
        browser.close()
    return shot


def main() -> int:
    import importlib.util

    if importlib.util.find_spec("playwright") is None:
        print("Install Playwright: pip install playwright && python -m playwright install chromium")
        return 1

    print(
        f"Waiting for API http://127.0.0.1:{API_PORT}/health and "
        f"dash http://127.0.0.1:{DASH_PORT}/ (model={DEMO_MODEL})...",
        flush=True,
    )
    if not wait_http(f"http://127.0.0.1:{API_PORT}/health"):
        print(f"API not reachable at http://127.0.0.1:{API_PORT}/health - start uvicorn first.")
        return 1
    if not wait_http(f"http://127.0.0.1:{DASH_PORT}/"):
        print(
            f"Dashboard not reachable at http://127.0.0.1:{DASH_PORT}/ - run: npm run dev in dashboard/"
        )
        return 1

    ho = (
        os.environ.get("ANIMA_DEMO_HEADLESS_ONLY")
        or os.environ.get("LLM_AFFECT_DEMO_HEADLESS_ONLY")
        or ""
    )
    headless_only = ho.strip().lower() in ("1", "true", "yes")
    modes = (True,) if headless_only else (False, True)
    for headless in modes:
        try:
            shot = run(headless=headless)
            mode = "headless" if headless else "headed"
            print(f"Demo OK ({mode}). Screenshot: {shot}")
            print("If headed browser flashed, you could watch the token stream live; screenshot is the fallback record.")
            return 0
        except Exception as e:
            print(f"Attempt headless={headless} failed: {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
