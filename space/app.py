"""HF Space: original Anima dashboard as the host ASGI app (ZeroGPU-safe)."""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

_DIST = Path(__file__).resolve().parent / "dashboard_dist"

# HF ignores launch(ssr_mode=False); env must be set before Gradio import.
os.environ["GRADIO_SSR_MODE"] = "false"
os.environ.setdefault("ANIMA_PUBLIC_MODE", "1")
os.environ.setdefault("ANIMA_FORCE_CPU", "1")
os.environ.setdefault("ANIMA_WARMUP_MODEL", "")
os.environ.setdefault("ANIMA_MAX_NEW_TOKENS", "64")
os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
os.environ.setdefault("GRADIO_SERVER_PORT", "7860")
os.environ["ANIMA_SERVE_DASHBOARD"] = str(_DIST)

_RELEASE = "https://github.com/Siddarthb07/Anima/releases/download/v2.0.0"
for _name in (
    "qwen2.5_0.5b_instruct_text.pt",
    "tinyllama_1.1b_chat_v1.0_text.pt",
    "tiny_random_gpt2_text.pt",
):
    try:
        from probes.zoo_io import ZOO_DIR

        ZOO_DIR.mkdir(parents=True, exist_ok=True)
        dest = ZOO_DIR / _name
        if not dest.exists():
            print(f"download {_name}", flush=True)
            urllib.request.urlretrieve(f"{_RELEASE}/{_name}", dest)
    except Exception as exc:
        print(f"skip {_name}: {exc}", flush=True)

# ZeroGPU Gradio launch probes localhost and can false-fail on Spaces.
try:
    from gradio import networking as _gradio_networking

    _gradio_networking.url_ok = lambda *args, **kwargs: True  # type: ignore[misc]
except Exception as _exc:
    print(f"Anima: could not patch gradio.networking.url_ok: {_exc}", flush=True)

import gradio as gr
import spaces
from gradio import routes as gr_routes

from api.server import app as anima_api


@spaces.GPU(duration=120)
def _zero_gpu_ping(x: str) -> str:
    return f"ok:{x[:32]}"


with gr.Blocks(title="Anima GPU bridge") as demo:
    gr.Markdown("ZeroGPU bridge at `/gradio`. Dashboard is the Space root `/`.")
    _in = gr.Textbox(value="ping")
    _out = gr.Textbox()
    _in.submit(_zero_gpu_ping, _in, _out)

demo.queue(default_concurrency_limit=2)

# Gradio's ZeroGPU startup path must call create_app so @spaces.GPU is detected.
# We return the FastAPI dashboard host with Gradio mounted under /gradio.
_ORIG_CREATE = gr_routes.App.create_app


def _create_app(demo_obj, **kwargs):
    gr_app = _ORIG_CREATE(demo_obj, **kwargs)
    already = any(getattr(r, "path", None) == "/gradio" for r in anima_api.router.routes)
    if not already:
        anima_api.mount("/gradio", gr_app)
    print("Anima: host=FastAPI dashboard+API; Gradio bridge at /gradio", flush=True)
    return anima_api


gr_routes.App.create_app = staticmethod(_create_app)  # type: ignore[method-assign]

_ORIG_BLOCKS_LAUNCH = gr.Blocks.launch


def _blocks_launch(self, *args, **kwargs):
    """HF calls class-level Blocks.launch; keep ZeroGPU detection + stay alive."""
    kwargs["server_name"] = "0.0.0.0"
    kwargs["server_port"] = int(os.environ.get("PORT") or "7860")
    kwargs["ssr_mode"] = False
    kwargs["share"] = False
    kwargs["inline"] = False
    # Must block (or sleep after) — prevent_thread_lock=True exits the Space with code 0.
    kwargs["prevent_thread_lock"] = False
    print("Anima: Gradio Blocks.launch (blocking, ZeroGPU-safe)", flush=True)
    try:
        return _ORIG_BLOCKS_LAUNCH(self, *args, **kwargs)
    except ValueError as exc:
        print(f"Anima: Gradio launch ValueError ({exc}); uvicorn fallback", flush=True)
        import uvicorn

        uvicorn.run(anima_api, host="0.0.0.0", port=int(os.environ.get("PORT") or "7860"))
        return None
    # If Gradio still returns, never exit.
    print("Anima: launch returned; keeping process alive", flush=True)
    while True:
        time.sleep(3600)


gr.Blocks.launch = _blocks_launch  # type: ignore[method-assign]
demo.launch = lambda *a, **k: _blocks_launch(demo, *a, **k)  # type: ignore[method-assign]

if __name__ == "__main__":
    _blocks_launch(demo)
