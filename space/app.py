"""HF Space: original Anima dashboard as the host ASGI app (ZeroGPU-safe)."""

from __future__ import annotations

import os
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
import uvicorn

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

# FastAPI hosts the React dashboard + API; Gradio is only the ZeroGPU bridge.
app = gr.mount_gradio_app(anima_api, demo, path="/gradio")
print("Anima: host=FastAPI dashboard+API; Gradio bridge at /gradio", flush=True)


def _serve(*_args, **_kwargs) -> None:
    port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or "7860")
    print(f"Anima: uvicorn serving dashboard on 0.0.0.0:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


# HF Spaces often calls Blocks.launch on the class, not the instance attribute.
_ORIG_BLOCKS_LAUNCH = gr.Blocks.launch


def _blocks_launch(self, *args, **kwargs):
    print("Anima: Blocks.launch → blocking uvicorn", flush=True)
    _serve()


gr.Blocks.launch = _blocks_launch  # type: ignore[method-assign]
demo.launch = _serve  # type: ignore[method-assign]

if __name__ == "__main__":
    _serve()
