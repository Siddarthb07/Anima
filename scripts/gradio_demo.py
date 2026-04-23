"""Lightweight Gradio demo (optional: pip install anima[gradio])."""

from __future__ import annotations

import os

from core.defaults import DEFAULT_CAUSAL_LM


def main() -> None:
    import gradio as gr
    import httpx

    api = os.environ.get("ANIMA_API_HTTP", "http://127.0.0.1:8010")

    def run(prompt: str, model: str, max_tokens: int):
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                f"{api}/generate",
                json={"model": model, "prompt": prompt, "max_new_tokens": max_tokens},
            )
            r.raise_for_status()
            data = r.json()
        text = "".join(t["token_text"] for t in data["tokens"])
        summary = data.get("summary", {})
        return text, summary

    gr.Interface(
        fn=run,
        inputs=[
            gr.Textbox(label="Prompt", value="Say hello in a few tokens."),
            gr.Textbox(label="Model", value=DEFAULT_CAUSAL_LM),
            gr.Slider(4, 64, value=16, step=1, label="Max tokens"),
        ],
        outputs=[gr.Textbox(label="Output"), gr.JSON(label="Summary")],
        title="Anima readout demo",
        description="Requires API running (anima api --port 8010).",
    ).launch()


if __name__ == "__main__":
    main()
