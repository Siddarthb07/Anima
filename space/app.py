"""HF Space: original Anima React dashboard (ZeroGPU-safe).

Normal ``demo.launch()`` so ZeroGPU detects ``@spaces.GPU``. After Gradio builds
its FastAPI app, we attach Anima API routes and serve ``dashboard_dist`` at ``/``.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

os.environ.setdefault("ANIMA_PUBLIC_MODE", "1")
os.environ.setdefault("ANIMA_FORCE_CPU", "1")
os.environ.setdefault("ANIMA_WARMUP_MODEL", "")
os.environ.setdefault("ANIMA_MAX_NEW_TOKENS", "64")

_DIST = Path(__file__).resolve().parent / "dashboard_dist"
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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from gradio import routes as gr_routes

# Dashboard mount happens in api.server when ANIMA_SERVE_DASHBOARD is set — but that
# would own ``/`` on a separate app. Here we graft routes onto Gradio's app instead.
os.environ.pop("ANIMA_SERVE_DASHBOARD", None)

from api import server as anima_server  # noqa: E402


@spaces.GPU(duration=120)
def _zero_gpu_ping(x: str) -> str:
    return f"ok:{x[:32]}"


with gr.Blocks(title="Anima") as demo:
    gr.Markdown(
        "### Anima\n\n"
        "If you still see this Gradio shell, hard-refresh — the React dashboard "
        "should replace `/`. Default model **Qwen2.5-0.5B**; also try **TinyLlama**."
    )
    _in = gr.Textbox(value="ping", label="ZeroGPU bridge")
    _out = gr.Textbox(label="status")
    _in.submit(_zero_gpu_ping, _in, _out)


_ORIG_CREATE = gr_routes.App.create_app


def _create_app(demo_obj, **kwargs):
    app = _ORIG_CREATE(demo_obj, **kwargs)

    # Anima API routes (HTTP + WebSocket). Lifespan/middleware stay on Gradio host.
    for route in list(anima_server.app.routes):
        app.routes.insert(0, route)

    if _DIST.is_dir():
        assets = _DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="anima-dash-assets")

        async def dashboard_root():
            return FileResponse(_DIST / "index.html")

        # Insert ahead of Gradio's UI route.
        app.add_api_route("/", dashboard_root, methods=["GET"])
        app.add_api_route("/index.html", dashboard_root, methods=["GET"])

    print("Anima: Gradio host + original dashboard assets + API routes", flush=True)
    return app


gr_routes.App.create_app = staticmethod(_create_app)  # type: ignore[method-assign]

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
