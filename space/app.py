"""Standalone Gradio demo for HF Space (college-apps v2.1).

Embedded inference — no separate FastAPI process. Set ANIMA_PUBLIC_MODE=1 on Space.
"""

from __future__ import annotations

import os
from typing import Any

# Public demo defaults (Space secrets should set ANIMA_PUBLIC_MODE=1).
os.environ.setdefault("ANIMA_PUBLIC_MODE", "1")

from alignment.tribe_encoder import TRIBEv2Encoder, tribe_seed
from core.defaults import HERO_DEMO_MODEL
from core.extractor import ActivationExtractor
from core.limits import PUBLIC_DEMO_MODELS, assert_model_allowed, clamp_max_new_tokens, validate_prompt
from probes.linear_probe import AffectProbe
from probes.zoo_io import load_probe_into, probe_slug, tribe_weights_path

HONESTY = (
    "**Instrumentation only** — valence/arousal readouts from hidden-state probes, "
    "not proof the model feels emotions. Text-emotion probes (GoEmotions); "
    "brain tier is synthetic_minimal where present. "
    "[Limits](https://github.com/Siddarthb07/Anima/blob/main/docs/USAGE_AND_LIMITATIONS.md)"
)

_cache: dict[str, tuple[ActivationExtractor, AffectProbe, dict[str, Any], TRIBEv2Encoder]] = {}


def _load_stack(model_name: str):
    assert_model_allowed(model_name)
    if model_name in _cache:
        return _cache[model_name]
    for name in list(_cache.keys()):
        if name != model_name:
            ex, _, _, _ = _cache.pop(name)
            try:
                ex.cleanup()
            except Exception:
                pass
    extractor = ActivationExtractor(model_name)
    probe = AffectProbe(extractor.hidden_dim, len(extractor.layer_indices))
    meta = load_probe_into(probe, model_name)
    slug = probe_slug(model_name)
    wpath = tribe_weights_path(slug)
    tribe = TRIBEv2Encoder(
        int(extractor.hidden_dim),
        seed=tribe_seed(model_name),
        weights_path=str(wpath) if wpath.exists() else None,
    )
    _cache[model_name] = (extractor, probe, meta, tribe)
    return _cache[model_name]


def run_readout(
    prompt: str,
    model: str,
    max_new_tokens: int,
    guard_mode: str,
    intervention_mode: str,
) -> tuple[str, dict[str, Any], str]:
    validate_prompt(prompt)
    max_new = clamp_max_new_tokens(int(max_new_tokens))
    extractor, probe, meta, _tribe = _load_stack(model.strip())
    rows = extractor.extract(
        prompt,
        max_new,
        probe=probe if intervention_mode == "dampen" else None,
        intervention_mode=intervention_mode,
    )
    valences: list[float] = []
    arousals: list[float] = []
    trace_lines: list[str] = []
    text_parts: list[str] = []
    for r in rows:
        aff = probe.predict(r["activations"])
        tok = str(r.get("token_text", ""))
        text_parts.append(tok)
        valences.append(float(aff["valence"]))
        arousals.append(float(aff["arousal"]))
        trace_lines.append(f"{tok!r}  v={aff['valence']:+.3f}  a={aff['arousal']:+.3f}")
    text = "".join(text_parts)
    summary: dict[str, Any] = {
        "model": model,
        "probe_origin": meta.get("probe_origin", "unknown"),
        "n_tokens": len(rows),
        "mean_valence": round(sum(valences) / len(valences), 4) if valences else 0.0,
        "mean_arousal": round(sum(arousals) / len(arousals), 4) if arousals else 0.0,
        "guard_mode": guard_mode,
        "intervention_mode": intervention_mode,
    }
    trace = "\n".join(trace_lines[:40])
    if len(trace_lines) > 40:
        trace += f"\n... ({len(trace_lines) - 40} more tokens)"
    return text, summary, trace


def build_ui():
    import gradio as gr

    model_choices = sorted(PUBLIC_DEMO_MODELS)
    default_model = HERO_DEMO_MODEL if HERO_DEMO_MODEL in PUBLIC_DEMO_MODELS else model_choices[0]

    with gr.Blocks(title="Anima — LLM affect readouts") as demo:
        gr.Markdown(f"# Anima readout demo\n\n{HONESTY}")
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                value="I'm thrilled — we finally shipped it and everything worked.",
                lines=3,
            )
        with gr.Row():
            model = gr.Dropdown(model_choices, value=default_model, label="Model (public allowlist)")
            max_tok = gr.Slider(4, 128, value=24, step=1, label="Max new tokens")
        with gr.Row():
            guard_mode = gr.Radio(["observe", "gate"], value="observe", label="Guard mode")
            intervention = gr.Radio(["none", "dampen"], value="none", label="Intervention")
        run_btn = gr.Button("Generate + readout", variant="primary")
        out_text = gr.Textbox(label="Generated text")
        out_summary = gr.JSON(label="Summary")
        out_trace = gr.Textbox(label="Per-token valence / arousal (first 40)", lines=12)

        run_btn.click(
            fn=run_readout,
            inputs=[prompt, model, max_tok, guard_mode, intervention],
            outputs=[out_text, out_summary, out_trace],
        )
        gr.Markdown(
            "Repo: [github.com/Siddarthb07/Anima](https://github.com/Siddarthb07/Anima) · "
            "Space: [huggingface.co/spaces/sidb078/Anima](https://huggingface.co/spaces/sidb078/Anima) · "
            "Hero model: **TinyLlama** (council 94, best prompt separation)."
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
