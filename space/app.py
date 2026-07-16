"""HF Space: original Anima dashboard as the host ASGI app (ZeroGPU-safe)."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

_DIST = Path(__file__).resolve().parent / "dashboard_dist"

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

import gradio as gr
import spaces
from gradio import routes as gr_routes

from api.server import app as anima_api  # mounts dashboard_dist when env set


@spaces.GPU(duration=120)
def _zero_gpu_ping(x: str) -> str:
    return f"ok:{x[:32]}"


with gr.Blocks(title="Anima GPU bridge") as demo:
    gr.Markdown("ZeroGPU bridge at `/gradio`. Dashboard is the Space root `/`.")
    _in = gr.Textbox(value="ping")
    _out = gr.Textbox()
    _in.submit(_zero_gpu_ping, _in, _out)


_ORIG_CREATE = gr_routes.App.create_app


def _create_app(demo_obj, **kwargs):
    """Build Gradio's internal app, mount it under Anima FastAPI, return Anima."""
    gr_app = _ORIG_CREATE(demo_obj, **kwargs)
    already = any(getattr(r, "path", None) == "/gradio" for r in anima_api.router.routes)
    if not already:
        anima_api.mount("/gradio", gr_app)
    print("Anima: host=FastAPI dashboard+API; Gradio bridge at /gradio", flush=True)
    return anima_api


gr_routes.App.create_app = staticmethod(_create_app)  # type: ignore[method-assign]

demo.queue(default_concurrency_limit=2)


def _serve_uvicorn() -> None:
    import uvicorn

    port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or "7860")
    print(f"Anima: uvicorn fallback on 0.0.0.0:{port}", flush=True)
    uvicorn.run(anima_api, host="0.0.0.0", port=port, log_level="info")


def _launch(*args, **kwargs):
    """HF / ZeroGPU call launch(); fall back to uvicorn if Gradio localhost check fails."""
    kwargs["server_name"] = "0.0.0.0"
    kwargs["server_port"] = int(os.environ.get("PORT") or "7860")
    kwargs["ssr_mode"] = False
    kwargs["share"] = False
    kwargs["inline"] = False
    try:
        return gr.blocks.Blocks.launch(demo, *args, **kwargs)
    except ValueError as exc:
        if "shareable link" not in str(exc).lower() and "localhost" not in str(exc).lower():
            raise
        print(f"Anima: Gradio launch blocked ({exc}); serving dashboard via uvicorn", flush=True)
        _serve_uvicorn()


demo.launch = _launch  # type: ignore[method-assign]

# Do not auto-call launch here — HF Spaces invokes demo.launch().
