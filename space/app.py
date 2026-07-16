"""HF Space entry: original Anima React dashboard + FastAPI.

Gradio Blocks exist only for ZeroGPU (@spaces.GPU). Public traffic is FastAPI
serving dashboard_dist/ (the real dashboard/ UI).
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("ANIMA_PUBLIC_MODE", "1")
os.environ.setdefault("ANIMA_FORCE_CPU", "1")
os.environ.setdefault("ANIMA_WARMUP_MODEL", "")
os.environ.setdefault("ANIMA_MAX_NEW_TOKENS", "64")

_DIST = Path(__file__).resolve().parent / "dashboard_dist"
os.environ["ANIMA_SERVE_DASHBOARD"] = str(_DIST)

_RELEASE = "https://github.com/Siddarthb07/Anima/releases/download/v2.0.0"
_PROBES = (
    "qwen2.5_0.5b_instruct_text.pt",
    "tinyllama_1.1b_chat_v1.0_text.pt",
    "tiny_random_gpt2_text.pt",
)


def _download_probes() -> None:
    from probes.zoo_io import ZOO_DIR

    ZOO_DIR.mkdir(parents=True, exist_ok=True)
    for name in _PROBES:
        dest = ZOO_DIR / name
        if dest.exists():
            continue
        try:
            print(f"download {name}", flush=True)
            urllib.request.urlretrieve(f"{_RELEASE}/{name}", dest)
        except Exception as exc:
            print(f"skip {name}: {exc}", flush=True)


_download_probes()

from api.server import app as fastapi_app  # noqa: E402

import gradio as gr  # noqa: E402
import spaces  # noqa: E402


@spaces.GPU(duration=120)
def _zero_gpu_ping(x: str) -> str:
    return f"ok:{x[:32]}"


with gr.Blocks(title="Anima GPU bridge") as demo:
    gr.Markdown("Dashboard is on the Space root URL. This `/gradio` page is the ZeroGPU bridge.")
    t = gr.Textbox(value="ping")
    o = gr.Textbox()
    t.submit(_zero_gpu_ping, t, o)


app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

_started = threading.Event()


def _serve() -> None:
    import uvicorn

    print("Anima: uvicorn dashboard+API on 0.0.0.0:7860", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")


def _launch_noop(*_a, **_k):
    """HF calls demo.launch() — start uvicorn once, then block."""
    if not _started.is_set():
        _started.set()
        threading.Thread(target=_serve, daemon=False).start()
        time.sleep(1.5)
    while True:
        time.sleep(3600)


demo.launch = _launch_noop  # type: ignore[method-assign]

# Start immediately so ASGI `app` export / early boot also works.
threading.Thread(target=_serve, daemon=True).start()
_started.set()

__all__ = ["app", "demo"]

if __name__ == "__main__":
    _serve()
