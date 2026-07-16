"""Standalone Gradio demo for Hugging Face Spaces.

HF Spaces: app.py at repo root, requirements.txt installs Anima from GitHub.
Probe weights download on first run from GitHub Release v2.0.0.
"""

from __future__ import annotations

import html
import os
import urllib.request
from typing import Any

# Public demo defaults on Space.
os.environ.setdefault("ANIMA_PUBLIC_MODE", "1")
_ON_HF_SPACE = bool(os.environ.get("SPACE_ID"))
# Local / CPU Spaces only — ZeroGPU allocates CUDA inside @spaces.GPU.
if not _ON_HF_SPACE:
    os.environ.setdefault("ANIMA_FORCE_CPU", "1")

_RELEASE_PROBE_ASSETS = (
    "tiny_random_gpt2_text.pt",
    "tinyllama_1.1b_chat_v1.0_text.pt",
    "qwen2.5_0.5b_instruct_text.pt",
)
_RELEASE_BASE = "https://github.com/Siddarthb07/Anima/releases/download/v2.0.0"

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

:root {
  --ink-950: #07090d;
  --ink-900: #0c1018;
  --ink-800: #151b27;
  --ink-700: #1e2738;
  --fog: #9aa8bc;
  --paper: #e8eef7;
  --cyan: #2de2e6;
  --rose: #ff6b8a;
  --amber: #f5c542;
  --line: rgba(232, 238, 247, 0.08);
}

.gradio-container {
  max-width: 1120px !important;
  margin: 0 auto !important;
  font-family: "Instrument Sans", system-ui, sans-serif !important;
  color: var(--paper) !important;
}

.gradio-container, .main, .wrap, .contain {
  background: transparent !important;
}

body, .app, .gradio-container {
  background:
    radial-gradient(ellipse 80% 50% at 10% -10%, rgba(45, 226, 230, 0.14), transparent 55%),
    radial-gradient(ellipse 60% 40% at 90% 0%, rgba(255, 107, 138, 0.10), transparent 50%),
    radial-gradient(ellipse 50% 60% at 50% 100%, rgba(245, 197, 66, 0.06), transparent 55%),
    linear-gradient(180deg, var(--ink-950) 0%, var(--ink-900) 40%, #0a0e15 100%) !important;
}

.anima-hero {
  position: relative;
  padding: 1.75rem 0 1.25rem;
  margin-bottom: 0.5rem;
  animation: fadeUp 0.7s ease-out both;
}

.anima-kicker {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--cyan);
  font-weight: 600;
  margin-bottom: 0.65rem;
}

.anima-kicker::before {
  content: "";
  width: 1.5rem;
  height: 2px;
  background: var(--cyan);
  box-shadow: 0 0 12px var(--cyan);
}

.anima-brand {
  font-family: "Syne", sans-serif;
  font-weight: 800;
  font-size: clamp(2.8rem, 8vw, 4.4rem);
  line-height: 0.92;
  letter-spacing: -0.04em;
  margin: 0;
  background: linear-gradient(135deg, #fff 20%, var(--cyan) 55%, var(--rose) 95%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  animation: brandIn 0.9s cubic-bezier(0.22, 1, 0.36, 1) both;
}

.anima-tagline {
  margin: 0.85rem 0 0;
  max-width: 42rem;
  font-size: 1.08rem;
  line-height: 1.6;
  color: var(--fog);
  animation: fadeUp 0.8s 0.12s ease-out both;
}

.anima-explain {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.85rem;
  margin: 1.15rem 0 0.35rem;
  animation: fadeUp 0.8s 0.18s ease-out both;
}

@media (max-width: 720px) {
  .anima-explain { grid-template-columns: 1fr; }
}

.explain-card {
  padding: 1rem 1.1rem;
  border-radius: 14px;
  border: 1px solid var(--line);
  background: rgba(21, 27, 39, 0.75);
}

.explain-card h3 {
  margin: 0 0 0.4rem;
  font-size: 0.78rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--cyan);
  font-weight: 700;
}

.explain-card p {
  margin: 0;
  font-size: 0.9rem;
  line-height: 1.5;
  color: #c5d0e0;
}

.explain-card.rose h3 { color: var(--rose); }
.explain-card.amber h3 { color: var(--amber); }

.model-cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.85rem;
  margin: 0.85rem 0 1.1rem;
  animation: fadeUp 0.8s 0.22s ease-out both;
}

@media (max-width: 720px) {
  .model-cards { grid-template-columns: 1fr; }
}

.model-card {
  padding: 1rem 1.1rem;
  border-radius: 14px;
  border: 1px solid var(--line);
  background: linear-gradient(145deg, rgba(30, 39, 56, 0.95), rgba(12, 16, 24, 0.9));
}

.model-card.hero {
  border-color: rgba(45, 226, 230, 0.35);
  box-shadow: 0 0 0 1px rgba(45, 226, 230, 0.08), 0 12px 32px rgba(0,0,0,0.25);
}

.model-card .badge {
  display: inline-block;
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-weight: 700;
  padding: 0.2rem 0.5rem;
  border-radius: 999px;
  margin-bottom: 0.45rem;
}

.model-card.hero .badge {
  background: rgba(45, 226, 230, 0.15);
  color: var(--cyan);
}

.model-card.council .badge {
  background: rgba(245, 197, 66, 0.12);
  color: var(--amber);
}

.model-card h3 {
  margin: 0 0 0.35rem;
  font-family: "Syne", sans-serif;
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--paper);
}

.model-card p {
  margin: 0;
  font-size: 0.86rem;
  line-height: 1.5;
  color: var(--fog);
}

.how-steps {
  margin: 0.5rem 0 1.1rem;
  padding: 1rem 1.15rem;
  border-radius: 14px;
  border: 1px solid var(--line);
  background: rgba(12, 16, 24, 0.65);
  animation: fadeUp 0.8s 0.26s ease-out both;
}

.how-steps ol {
  margin: 0.4rem 0 0;
  padding-left: 1.2rem;
  color: #c5d0e0;
  font-size: 0.9rem;
  line-height: 1.55;
}

.how-steps strong { color: var(--paper); }

.anima-limit {
  margin-top: 0.85rem;
  padding: 0.65rem 0.85rem;
  border-left: 2px solid var(--amber);
  background: rgba(245, 197, 66, 0.06);
  color: #d5c9a0;
  font-size: 0.86rem;
  line-height: 1.45;
  animation: fadeUp 0.8s 0.2s ease-out both;
}

.anima-limit a { color: var(--cyan); }

.anima-panel {
  border: 1px solid var(--line) !important;
  background: rgba(12, 16, 24, 0.72) !important;
  backdrop-filter: blur(12px);
  border-radius: 16px !important;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35) !important;
}

label, .label-wrap span {
  font-size: 0.78rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--fog) !important;
  font-weight: 600 !important;
}

textarea, input, .wrap-inner, .secondary-wrap {
  background: var(--ink-800) !important;
  border-color: var(--line) !important;
  color: var(--paper) !important;
  border-radius: 12px !important;
  font-family: "Instrument Sans", sans-serif !important;
}

textarea:focus, input:focus {
  border-color: rgba(45, 226, 230, 0.45) !important;
  box-shadow: 0 0 0 3px rgba(45, 226, 230, 0.12) !important;
}

button.primary, .primary {
  background: linear-gradient(135deg, var(--cyan), #1ab8c4) !important;
  color: #041016 !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  border: none !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 28px rgba(45, 226, 230, 0.28) !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}

button.primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 32px rgba(45, 226, 230, 0.4) !important;
}

.examples {
  margin-top: 0.35rem !important;
}
.examples table {
  background: transparent !important;
  border: none !important;
}
.examples button, .examples td {
  background: var(--ink-800) !important;
  border: 1px solid var(--line) !important;
  color: var(--fog) !important;
  border-radius: 999px !important;
  font-size: 0.82rem !important;
}

.meter-board {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin: 0.25rem 0 1rem;
  animation: fadeUp 0.45s ease-out both;
}

@media (max-width: 640px) {
  .meter-board { grid-template-columns: 1fr; }
}

.meter {
  padding: 1rem 1.1rem 1.15rem;
  border-radius: 14px;
  border: 1px solid var(--line);
  background: linear-gradient(160deg, rgba(30, 39, 56, 0.9), rgba(12, 16, 24, 0.95));
}

.meter-label {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.55rem;
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--fog);
  font-weight: 600;
}

.meter-value {
  font-family: "JetBrains Mono", monospace;
  font-size: 1.35rem;
  letter-spacing: -0.02em;
  color: var(--paper);
}

.meter-track {
  height: 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  overflow: hidden;
  position: relative;
}

.meter-fill {
  height: 100%;
  border-radius: 999px;
  width: 0%;
  transition: width 0.65s cubic-bezier(0.22, 1, 0.36, 1);
}

.meter-fill.valence {
  background: linear-gradient(90deg, #ff6b8a, #2de2e6);
  box-shadow: 0 0 16px rgba(45, 226, 230, 0.35);
}

.meter-fill.arousal {
  background: linear-gradient(90deg, #3b82f6, #f5c542);
  box-shadow: 0 0 16px rgba(245, 197, 66, 0.35);
}

.meter-hint {
  margin-top: 0.45rem;
  font-size: 0.78rem;
  color: #7d8aa0;
}

.token-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin-top: 0.75rem;
  max-height: 9.5rem;
  overflow-y: auto;
  padding: 0.15rem;
}

.tok {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.72rem;
  padding: 0.28rem 0.45rem;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.03);
  color: var(--paper);
  white-space: pre;
}

.anima-foot {
  margin-top: 1.25rem;
  padding-top: 1rem;
  border-top: 1px solid var(--line);
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem 1.25rem;
  font-size: 0.82rem;
  color: var(--fog);
  animation: fadeUp 0.7s 0.25s ease-out both;
}

.anima-foot a {
  color: var(--cyan);
  text-decoration: none;
  font-weight: 600;
}
.anima-foot a:hover { text-decoration: underline; }

.footer { display: none !important; }

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes brandIn {
  from { opacity: 0; transform: translateY(18px) scale(0.98); filter: blur(4px); }
  to { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
}
"""


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
from core.defaults import COUNCIL_BEST_MODEL, DEFAULT_CAUSAL_LM, HERO_DEMO_MODEL
from core.extractor import ActivationExtractor
from core.limits import PUBLIC_DEMO_MODELS, assert_model_allowed, clamp_max_new_tokens, validate_prompt
from probes.linear_probe import AffectProbe
from probes.zoo_io import ZOO_DIR, load_probe_into, probe_slug, tribe_weights_path

_cache: dict[str, tuple[ActivationExtractor, AffectProbe, dict[str, Any], TRIBEv2Encoder]] = {}

EXAMPLE_PROMPTS = [
    ["I'm thrilled — we finally shipped it and everything worked."],
    ["I keep replaying that conversation and I can't shake the dread."],
    ["The lab was quiet. Numbers on the screen didn't move."],
    ["Honestly? I'm fine. Totally fine. Don't worry about it."],
]

QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TINYLLAMA_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _available_models() -> list[str]:
    """Prefer Qwen + TinyLlama first; only list models with probe weights on disk."""
    slug_map = {
        "hf-internal-testing/tiny-random-gpt2": "tiny_random_gpt2",
        TINYLLAMA_ID: "tinyllama_1.1b_chat_v1.0",
        QWEN_ID: "qwen2.5_0.5b_instruct",
    }
    preferred = [QWEN_ID, TINYLLAMA_ID, "hf-internal-testing/tiny-random-gpt2"]
    out: list[str] = []
    for mid in preferred:
        if mid not in PUBLIC_DEMO_MODELS:
            continue
        slug = slug_map.get(mid, probe_slug(mid))
        if (ZOO_DIR / f"{slug}_text.pt").exists():
            out.append(mid)
    for mid in sorted(PUBLIC_DEMO_MODELS):
        if mid in out:
            continue
        slug = slug_map.get(mid, probe_slug(mid))
        if (ZOO_DIR / f"{slug}_text.pt").exists():
            out.append(mid)
    return out or [DEFAULT_CAUSAL_LM]


def _default_model(models: list[str]) -> str:
    """Default = Qwen hero; then TinyLlama (council best); then anything available."""
    for mid in (HERO_DEMO_MODEL, COUNCIL_BEST_MODEL, QWEN_ID, TINYLLAMA_ID):
        if mid in models:
            return mid
    return models[0]


def _short_model_label(mid: str) -> str:
    aliases = {
        "hf-internal-testing/tiny-random-gpt2": "tiny-gpt2 · smoke test only",
        TINYLLAMA_ID: "TinyLlama · best council score (94)",
        QWEN_ID: "Qwen2.5-0.5B · demo hero",
    }
    return aliases.get(mid, mid)


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


def _valence_word(v: float) -> str:
    if v >= 0.45:
        return "bright / positive"
    if v >= 0.15:
        return "mildly positive"
    if v <= -0.45:
        return "dark / negative"
    if v <= -0.15:
        return "mildly negative"
    return "near-neutral"


def _arousal_word(a: float) -> str:
    if a >= 0.55:
        return "high activation"
    if a >= 0.25:
        return "elevated"
    if a <= -0.15:
        return "low / calm"
    return "moderate"


def _meter_html(
    mean_v: float,
    mean_a: float,
    tokens: list[tuple[str, float, float]],
) -> str:
    # Map [-1,1] approx to bar width; clamp for display.
    def pct(x: float) -> float:
        return max(0.0, min(100.0, (float(x) + 1.0) * 50.0))

    chips = []
    for tok, v, a in tokens[:48]:
        shown = tok if tok.strip() else "␣"
        tint = int(max(0, min(100, (v + 1) * 50)))
        chips.append(
            f'<span class="tok" title="v={v:+.3f} a={a:+.3f}" '
            f'style="border-color:hsla({180 + tint * 0.6:.0f},70%,55%,0.35)">'
            f"{html.escape(shown)}</span>"
        )
    strip = "".join(chips) if chips else '<span class="tok">—</span>'
    return f"""
<div class="meter-board">
  <div class="meter">
    <div class="meter-label"><span>Valence (pleasant ↔ unpleasant)</span><span class="meter-value">{mean_v:+.3f}</span></div>
    <div class="meter-track"><div class="meter-fill valence" style="width:{pct(mean_v):.1f}%"></div></div>
    <div class="meter-hint">{_valence_word(mean_v)} · negative ← 0 → positive</div>
  </div>
  <div class="meter">
    <div class="meter-label"><span>Arousal (calm ↔ activated)</span><span class="meter-value">{mean_a:+.3f}</span></div>
    <div class="meter-track"><div class="meter-fill arousal" style="width:{pct(mean_a):.1f}%"></div></div>
    <div class="meter-hint">{_arousal_word(mean_a)} · how “charged” the internal state looks</div>
  </div>
</div>
<div class="meter-label" style="margin-bottom:0.35rem">Token trail — each chip is one generated token (hover for v / a)</div>
<div class="token-strip">{strip}</div>
"""


def _empty_meters() -> str:
    return _meter_html(0.0, 0.0, [])


def run_readout(
    prompt: str,
    model: str,
    max_new_tokens: int,
    guard_mode: str,
    intervention_mode: str,
) -> tuple[str, str, dict[str, Any], str]:
    if not prompt or not prompt.strip():
        return _empty_meters(), "", {"error": "empty prompt"}, ""
    # Dropdown may show short labels — map back to HF ids.
    label_to_id = {_short_model_label(m): m for m in _available_models()}
    model_id = label_to_id.get(model, model)
    validate_prompt(prompt)
    max_new = clamp_max_new_tokens(int(max_new_tokens))
    try:
        extractor, probe, meta, _tribe = _load_stack(model_id.strip())
        rows = extractor.extract(
            prompt,
            max_new,
            probe=probe if intervention_mode == "dampen" else None,
            intervention_mode=intervention_mode,
        )
    except Exception as exc:
        return _empty_meters(), "", {"error": str(exc)}, ""
    valences: list[float] = []
    arousals: list[float] = []
    trace_lines: list[str] = []
    text_parts: list[str] = []
    token_va: list[tuple[str, float, float]] = []
    for r in rows:
        aff = probe.predict(r["activations"])
        tok = str(r.get("token_text", ""))
        text_parts.append(tok)
        v = float(aff["valence"])
        a = float(aff["arousal"])
        valences.append(v)
        arousals.append(a)
        token_va.append((tok, v, a))
        trace_lines.append(f"{tok!r}  v={v:+.3f}  a={a:+.3f}")
    text = "".join(text_parts)
    mean_v = round(sum(valences) / len(valences), 4) if valences else 0.0
    mean_a = round(sum(arousals) / len(arousals), 4) if arousals else 0.0
    summary: dict[str, Any] = {
        "model": model_id,
        "probe_origin": meta.get("probe_origin", "unknown"),
        "n_tokens": len(rows),
        "mean_valence": mean_v,
        "mean_arousal": mean_a,
        "guard_mode": guard_mode,
        "intervention_mode": intervention_mode,
        "readout": "instrumentation — not subjective experience",
    }
    trace = "\n".join(trace_lines[:40])
    if len(trace_lines) > 40:
        trace += f"\n... ({len(trace_lines) - 40} more tokens)"
    return _meter_html(mean_v, mean_a, token_va), text, summary, trace


# ZeroGPU Spaces require at least one @spaces.GPU entrypoint at import time.
if _ON_HF_SPACE:
    try:
        import spaces

        run_readout = spaces.GPU(duration=120)(run_readout)  # type: ignore[misc]
    except Exception as exc:
        print(f"Anima: spaces.GPU wrap skipped: {exc}", flush=True)


def build_ui():
    import gradio as gr

    model_ids = _available_models()
    default_id = _default_model(model_ids)
    model_labels = [_short_model_label(m) for m in model_ids]
    default_label = _short_model_label(default_id)

    theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#ecfeff",
            c100="#cffafe",
            c200="#a5f3fc",
            c300="#67e8f9",
            c400="#22d3ee",
            c500="#2de2e6",
            c600="#0891b2",
            c700="#0e7490",
            c800="#155e75",
            c900="#164e63",
            c950="#083344",
        ),
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Instrument Sans"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#07090d",
        body_text_color="#e8eef7",
        block_background_fill="#0c1018",
        block_border_color="rgba(232,238,247,0.08)",
        block_label_text_color="#9aa8bc",
        input_background_fill="#151b27",
        button_primary_background_fill="#2de2e6",
        button_primary_text_color="#041016",
    )

    hero = """
<div class="anima-hero">
  <div class="anima-kicker">What is this?</div>
  <h1 class="anima-brand">ANIMA</h1>
  <p class="anima-tagline">
    <strong style="color:#e8eef7">Anima is not a chatbot that “has feelings.”</strong>
    It is a research instrument: while a language model generates text, we attach tiny
    linear probes to its hidden states and read two continuous scores —
    <strong style="color:#2de2e6">valence</strong> and
    <strong style="color:#ff6b8a">arousal</strong> —
    mapped from emotion-label data (GoEmotions). Think oscilloscope for internal geometry,
    not a claim about consciousness.
  </p>

  <div class="anima-explain">
    <div class="explain-card">
      <h3>Valence</h3>
      <p>
        Pleasant ↔ unpleasant axis. Positive means the probe reads the activation pattern
        as closer to “good / affirming” labels; negative as closer to “bad / aversive.”
        Scale is roughly −1 to +1 (instrument units, not human ratings).
      </p>
    </div>
    <div class="explain-card rose">
      <h3>Arousal</h3>
      <p>
        Calm ↔ activated axis. High arousal is the probe’s estimate of intensity /
        activation in the same geometry (excited, tense, alarmed territory), not heart rate.
        Low arousal looks quieter / flatter.
      </p>
    </div>
  </div>

  <div class="how-steps">
    <h3 style="margin:0;font-size:0.78rem;letter-spacing:0.14em;text-transform:uppercase;color:#f5c542;font-weight:700">
      How to use this demo
    </h3>
    <ol>
      <li><strong>Pick a model</strong> — default is <strong>Qwen2.5-0.5B</strong> (demo hero). Also try <strong>TinyLlama</strong> (highest council score).</li>
      <li><strong>Enter a prompt</strong> with emotional charge (or use an example below).</li>
      <li>Hit <strong>Run readout</strong> — the model generates tokens; meters update from probe readouts on each token’s hidden state.</li>
      <li><strong>Guard / Intervention</strong> are optional reliability knobs (observe vs gate abstention; optional dampen steering) — advanced, not required to understand the meters.</li>
    </ol>
  </div>

  <div class="model-cards">
    <div class="model-card hero">
      <span class="badge">Demo hero · default</span>
      <h3>Qwen2.5-0.5B-Instruct</h3>
      <p>
        Compact instruct model — default on this Space so visitors land on a real
        checkpoint (not a random smoke toy). Good for seeing clear valence / arousal
        movement on everyday emotional prompts.
      </p>
    </div>
    <div class="model-card council">
      <span class="badge">Council best · score 94</span>
      <h3>TinyLlama-1.1B-Chat</h3>
      <p>
        Strongest Anima council score in our CPU-tier rollup (best prompt separation).
        Switch to it in the dropdown to compare the same prompt on the benchmark leader.
      </p>
    </div>
  </div>

  <div class="anima-limit">
    Honest limit: these are <strong>constructed measurements</strong> from trained probes —
    not proof the model “feels” anything.
    <a href="https://github.com/Siddarthb07/Anima/blob/main/docs/USAGE_AND_LIMITATIONS.md" target="_blank" rel="noopener">Read usage &amp; limitations</a>
  </div>
</div>
"""

    with gr.Blocks(title="Anima — LLM affect readouts", theme=theme, css=_CSS) as ui:
        gr.HTML(hero)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                with gr.Group(elem_classes=["anima-panel"]):
                    prompt = gr.Textbox(
                        label="Your prompt (the text the model continues from)",
                        value=EXAMPLE_PROMPTS[0][0],
                        lines=4,
                        placeholder="Type something emotionally charged…",
                    )
                    gr.Examples(
                        examples=EXAMPLE_PROMPTS,
                        inputs=prompt,
                        label="Example prompts (different emotional charges)",
                    )
                    with gr.Row():
                        model = gr.Dropdown(
                            choices=model_labels,
                            value=default_label,
                            label="Which model to instrument",
                            info="Qwen = demo default · TinyLlama = best council score",
                        )
                        max_tok = gr.Slider(
                            4,
                            64,
                            value=24,
                            step=1,
                            label="How many new tokens to generate",
                        )
                    with gr.Row():
                        guard_mode = gr.Radio(
                            ["observe", "gate"],
                            value="observe",
                            label="Guard (observe = always show labels; gate = hide when unstable)",
                        )
                        intervention = gr.Radio(
                            ["none", "dampen"],
                            value="none",
                            label="Intervention (optional steering if valence swings)",
                        )
                    run_btn = gr.Button("Run readout →", variant="primary", size="lg")
            with gr.Column(scale=6):
                with gr.Group(elem_classes=["anima-panel"]):
                    gr.HTML(
                        "<p style='margin:0 0 0.75rem;color:#9aa8bc;font-size:0.9rem;line-height:1.45'>"
                        "<strong style='color:#e8eef7'>Live readout</strong> — after you run, "
                        "bars show mean valence &amp; arousal across generated tokens; "
                        "chips below are individual tokens."
                        "</p>"
                    )
                    meters = gr.HTML(value=_empty_meters(), label="Affect meters")
                    out_text = gr.Textbox(
                        label="What the model generated",
                        lines=3,
                        info="Plain continuation text from the chosen LM",
                    )
                    with gr.Accordion("Advanced: JSON summary & per-token numbers", open=False):
                        out_summary = gr.JSON(label="Machine-readable summary")
                        out_trace = gr.Textbox(label="Per-token valence / arousal log", lines=8)

        run_btn.click(
            fn=run_readout,
            inputs=[prompt, model, max_tok, guard_mode, intervention],
            outputs=[meters, out_text, out_summary, out_trace],
        )

        gr.HTML(
            """
<div class="anima-foot">
  <a href="https://github.com/Siddarthb07/Anima" target="_blank" rel="noopener">Source on GitHub</a>
  <span>Probes · FastAPI · React dashboard</span>
  <span>Default model: Qwen2.5-0.5B · Compare with TinyLlama (council 94)</span>
</div>
"""
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
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", "7860"))),
    )
