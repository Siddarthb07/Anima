"""HF Space entry: original Anima React dashboard + FastAPI on one ASGI app.

Gradio is mounted only so ZeroGPU detects @spaces.GPU. The public UI is the
real dashboard/ build served by FastAPI (same as local docker-compose).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

os.environ.setdefault("ANIMA_PUBLIC_MODE", "1")
# ZeroGPU only wraps @spaces.GPU Gradio fns — dashboard API runs on CPU.
os.environ.setdefault("ANIMA_FORCE_CPU", "1")
os.environ.setdefault("ANIMA_WARMUP_MODEL", "")  # load Qwen on first request (faster boot)
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

# Import after ANIMA_SERVE_DASHBOARD is set so StaticFiles mount engages.
from api.server import app as fastapi_app  # noqa: E402

import gradio as gr  # noqa: E402
import spaces  # noqa: E402


@spaces.GPU(duration=120)
def _zero_gpu_ping(x: str) -> str:
    """Required for ZeroGPU Spaces; dashboard inference uses the FastAPI routes."""
    return f"ok:{x[:32]}"


with gr.Blocks(title="Anima GPU bridge") as demo:
    gr.Markdown(
        "Anima dashboard is at **/** (this `/gradio` page is only the ZeroGPU bridge)."
    )
    inp = gr.Textbox(value="ping", label="GPU bridge")
    out = gr.Textbox(label="status")
    inp.submit(_zero_gpu_ping, inp, out)

# Public ASGI app: dashboard + API at /, Gradio bridge at /gradio
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")


def _launch_asgi(*_a, **_k):
    """HF Gradio SDK calls demo.launch(); serve FastAPI+dashboard on the public port."""
    import uvicorn

    # HF Spaces expose 7860; ignore Gradio's internal 7861 default when already bound.
    port = int(os.environ.get("PORT") or "7860")
    print(f"Anima: serving dashboard+API on 0.0.0.0:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


demo.launch = _launch_asgi  # type: ignore[method-assign]

# HF Gradio SDK also looks for `demo`.
__all__ = ["app", "demo"]

if __name__ == "__main__":
    _launch_asgi()
