"""HF Space: original Anima React dashboard (ZeroGPU-safe Gradio launch)."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

os.environ.setdefault("ANIMA_PUBLIC_MODE", "1")
os.environ.setdefault("ANIMA_FORCE_CPU", "1")
os.environ.setdefault("ANIMA_WARMUP_MODEL", "")
os.environ.setdefault("ANIMA_MAX_NEW_TOKENS", "64")
# Gradio host serves the dashboard; do not double-mount SPA on the Anima app object.
os.environ.pop("ANIMA_SERVE_DASHBOARD", None)

_DIST = Path(__file__).resolve().parent / "dashboard_dist"
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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from gradio import routes as gr_routes
from starlette.routing import Mount, Route, WebSocketRoute

from api import server as anima_server


@spaces.GPU(duration=120)
def _zero_gpu_ping(x: str) -> str:
    return f"ok:{x[:32]}"


with gr.Blocks(title="Anima") as demo:
    gr.Markdown("Anima ZeroGPU bridge — dashboard should load at `/`.")
    _in = gr.Textbox(value="ping")
    _out = gr.Textbox()
    _in.submit(_zero_gpu_ping, _in, _out)


_ORIG_CREATE = gr_routes.App.create_app


def _create_app(demo_obj, **kwargs):
    app = _ORIG_CREATE(demo_obj, **kwargs)

    # Prefer Anima API + dashboard over Gradio's default `/` UI.
    grafted: list = []
    for route in anima_server.app.router.routes:
        if isinstance(route, (Route, WebSocketRoute, Mount)):
            grafted.append(route)
    # Dashboard static assets + index (must be before Gradio catch-alls in practice
    # via insert at front).
    if _DIST.is_dir():
        assets = _DIST / "assets"
        if assets.is_dir():
            grafted.insert(0, Mount("/assets", StaticFiles(directory=str(assets))))

        async def dash_index(_request):
            return FileResponse(_DIST / "index.html")

        grafted.insert(0, Route("/", dash_index, methods=["GET"]))
        grafted.insert(0, Route("/index.html", dash_index, methods=["GET"]))

    app.router.routes = grafted + list(app.router.routes)
    print(f"Anima: grafted {len(grafted)} dashboard/API routes ahead of Gradio", flush=True)
    return app


gr_routes.App.create_app = staticmethod(_create_app)  # type: ignore[method-assign]

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
