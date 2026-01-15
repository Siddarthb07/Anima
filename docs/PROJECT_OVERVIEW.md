# Anima — project overview

This note is for anyone opening the repo for the first time: what each part does, and where the scientific claims stop.

## Big picture

Anima attaches **forward hooks** to a Hugging Face **causal LM**, records hidden states at configured layers, and passes them through:

1. A small **probe network** (`probes/linear_probe.py`) that outputs scalars we call **valence**, **arousal**, and **uncertainty** (probe head — not a clinical score).
2. Extra **uncertainty-style diagnostics** from logits and attention (see `core/extractor.py`).
3. An optional **TRIBEv2-style surrogate** (`alignment/tribe_encoder.py`) that maps the **same** layer activations to named ROI-like axes for the dashboard.

The **API** (`api/server.py`) exposes synchronous `/generate` and a **WebSocket** stream that emits one message per token and a final `done` payload. The **dashboard** is a plain React app that consumes that stream.

---

## Repository layout

| Directory | Role |
|-----------|------|
| `core/` | Layer map (`layer_config.py`), hook installation, `extract` / streaming iteration, suppression helpers. |
| `probes/` | `AffectProbe`, training (`train.py`), validation helpers; trained weights go under `probes/zoo/*.pt`. |
| `alignment/` | Narratives / atlas-oriented utilities, word–token alignment, **TRIBEv2 surrogate encoder** (linear sketch, not fMRI decoding). |
| `api/` | FastAPI app, Pydantic schemas, WebSocket protocol. |
| `cli/` | Smoke commands and optional `api` launcher. |
| `dashboard/` | Vite + React UI. |
| `tests/` | Pytest suite; HF-heavy tests gated behind `RUN_HF_TESTS=1`. |

---

## Probes and the “zoo”

If **`probes/zoo/<model_slug>.pt`** exists for your model, the API loads those weights. The slug is derived from the Hugging Face id (last path segment, normalized).

If the file **does not exist**, the probe is **randomly initialized**. That’s intentional for plumbing and CI: the pipeline runs end-to-end, but **readouts are not calibrated** to human affect or neural data. You’ll often see **neutral region labels** because the thresholds in `label_region()` never fire.

To get meaningful quadrants, train with your own targets via `probes/train.py` (and honest evaluation), then ship the checkpoint into `probes/zoo/`.

---

## TRIBEv2 surrogate vs real TRIBE

The dashboard’s **TRIBEv2 surrogate** block is a **deterministic, seeded linear map** from probed activations to ROI scalars and a derived valence/arousal sketch. It exists so the UI can show a **second pathway** tied to the same tensors as the probe.

It is **not** claiming to reproduce the TRIBE paper’s voxel decoder or measured fMRI. Don’t cite it as brain imaging evidence — cite it as **surrogate visualization** if you cite it at all.

---

## Language and framing

When you write about Anima externally:

- Prefer **readout**, **internal state**, **geometry**, **population-level analogy**.
- Avoid stating that the model **feels** an emotion as fact.

The UI copy tries to follow that; forks should too. See [USAGE_AND_LIMITATIONS.md](USAGE_AND_LIMITATIONS.md).

---

## Configuration knobs worth knowing

- **Default LM:** `core/defaults.py` → `DEFAULT_CAUSAL_LM` (tiny model for reliability).
- **Supported architectures / layer indices:** `core/layer_config.py`.
- **Dashboard env:** `dashboard/.env.example` → proxy target, optional direct WebSocket, default model id.

---

## Beta expectations

Interfaces and defaults may change between beta tags. If you depend on a stable JSON shape for `/generate` or the WebSocket, pin a commit or tag once releases exist.
