"""HF Space: original Anima React dashboard (ZeroGPU-safe Gradio launch).

``demo.launch()`` stays normal so ZeroGPU sees ``@spaces.GPU``. Gradio is mounted
under ``/gradio``; the Space root is the real dashboard + FastAPI API.
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
from gradio import routes as gr_routes

# Import after ANIMA_SERVE_DASHBOARD so API mounts dashboard_dist at /.
from api.server import app as anima_api  # noqa: E402


@spaces.GPU(duration=120)
def _zero_gpu_ping(x: str) -> str:
    return f"ok:{x[:32]}"


with gr.Blocks(title="Anima GPU bridge") as demo:
    gr.Markdown("ZeroGPU bridge only — use the Space **root URL** for the Anima dashboard.")
    _in = gr.Textbox(value="ping", label="ping")
    _out = gr.Textbox(label="status")
    _in.submit(_zero_gpu_ping, _in, _out)


_ORIG_CREATE = gr_routes.App.create_app


def _create_app(demo_obj, **kwargs):
    """Use Anima FastAPI (dashboard + API) as the host; Gradio under /gradio."""
    print("Anima: create_app → mount Gradio at /gradio on FastAPI dashboard app", flush=True)
    return gr.mount_gradio_app(anima_api, demo_obj, path="/gradio")


gr_routes.App.create_app = staticmethod(_create_app)  # type: ignore[method-assign]

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
