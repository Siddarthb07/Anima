"""Standalone Gradio demo for Hugging Face Spaces.

HF Spaces: app.py at repo root, requirements.txt installs Anima from GitHub.
Probe weights download on first run from GitHub Release v2.0.0.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

# Public demo defaults on Space.
os.environ.setdefault("ANIMA_PUBLIC_MODE", "1")
os.environ.setdefault("ANIMA_FORCE_CPU", "1")

_ON_HF_SPACE = bool(os.environ.get("SPACE_ID"))

_RELEASE_PROBE_ASSETS = (
    "tiny_random_gpt2_text.pt",
    "tinyllama_1.1b_chat_v1.0_text.pt",
    "qwen2.5_0.5b_instruct_text.pt",
)
_RELEASE_BASE = "https://github.com/Siddarthb07/Anima/releases/download/v2.0.0"


def _ensure_probe_weights() -> None:
    """Download minimal Release probes if missing (HF Space has no .pt in pip wheel)."""
    from probes.zoo_io import ZOO_DIR

    ZOO_DIR.mkdir(parents=True, exist_ok=True)
    missing = [n for n in _RELEASE_PROBE_ASSETS if not (ZOO_DIR / n).exists()]
    if not missing:
        return
    print(f"Anima: downloading {len(missing)} probe checkpoint(s)...", flush=True)
    for name in missing:
        dest = ZOO_DIR / name
        url = f"{_RELEASE_BASE}/{name}"
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  ok {name}", flush=True)
        except Exception as exc:
            print(f"  skip {name}: {exc}", flush=True)


_ensure_probe_weights()

from alignment.tribe_encoder import TRIBEv2Encoder, tribe_seed
from core.defaults import DEFAULT_CAUSAL_LM, HERO_DEMO_MODEL
from core.extractor import ActivationExtractor
from core.limits import PUBLIC_DEMO_MODELS, assert_model_allowed, clamp_max_new_tokens, validate_prompt
from probes.linear_probe import AffectProbe
from probes.zoo_io import ZOO_DIR, load_probe_into, probe_slug, tribe_weights_path

HONESTY = (
    "**Instrumentation only** — valence/arousal readouts from hidden-state probes, "
    "not proof the model feels emotions. Text-emotion probes (GoEmotions). "
    "[Limits](https://github.com/Siddarthb07/Anima/blob/main/docs/USAGE_AND_LIMITATIONS.md)"
)

_cache: dict[str, tuple[ActivationExtractor, AffectProbe, dict[str, Any], TRIBEv2Encoder]] = {}


def _available_models() -> list[str]:
    """Only list models whose text probe file exists (avoid HF 400 on missing zoo)."""
    slug_map = {
        "hf-internal-testing/tiny-random-gpt2": "tiny_random_gpt2",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "tinyllama_1.1b_chat_v1.0",
        "Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5_0.5b_instruct",
    }
    out = []
    for mid in sorted(PUBLIC_DEMO_MODELS):
        slug = slug_map.get(mid, probe_slug(mid))
        if (ZOO_DIR / f"{slug}_text.pt").exists():
            out.append(mid)
    return out or [DEFAULT_CAUSAL_LM]


def _default_model(models: list[str]) -> str:
    if _ON_HF_SPACE:
        # Free CPU tier: start with tiny; user can switch to TinyLlama if RAM allows.
        if DEFAULT_CAUSAL_LM in models:
            return DEFAULT_CAUSAL_LM
    if HERO_DEMO_MODEL in models:
        return HERO_DEMO_MODEL
    return models[0]


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
    if not prompt or not prompt.strip():
        return "", {"error": "empty prompt"}, ""
    validate_prompt(prompt)
    max_new = clamp_max_new_tokens(int(max_new_tokens))
    try:
        extractor, probe, meta, _tribe = _load_stack(model.strip())
        rows = extractor.extract(
            prompt,
            max_new,
            probe=probe if intervention_mode == "dampen" else None,
            intervention_mode=intervention_mode,
        )
    except Exception as exc:
        return "", {"error": str(exc)}, ""
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

    model_choices = _available_models()
    default_model = _default_model(model_choices)
    space_note = (
        "\n\n*HF Space: default is **tiny-random-gpt2** for free CPU RAM. "
        "Switch to **TinyLlama** for hero readouts if the Space stays up.*"
        if _ON_HF_SPACE
        else ""
    )

    with gr.Blocks(title="Anima — LLM affect readouts") as ui:
        gr.Markdown(f"# Anima readout demo\n\n{HONESTY}{space_note}")
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                value="I'm thrilled — we finally shipped it and everything worked.",
                lines=3,
            )
        with gr.Row():
            model = gr.Dropdown(model_choices, value=default_model, label="Model")
            max_tok = gr.Slider(4, 64, value=16, step=1, label="Max new tokens")
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
            "[GitHub](https://github.com/Siddarthb07/Anima) · "
            "Hero: **TinyLlama** (council 94) · Demo Space: **sidb078/Anima**"
        )
    return ui


# HF Gradio SDK loads this symbol — build lazily so import errors surface clearly.
try:
    demo = build_ui()
except Exception as exc:
    import traceback

    print("Anima Space UI failed to build:", flush=True)
    traceback.print_exc()
    import gradio as gr

    demo = gr.Interface(
        fn=lambda: f"UI build failed: {exc}",
        inputs=[],
        outputs="text",
        title="Anima (error)",
    )

if __name__ == "__main__":
    demo.launch()
