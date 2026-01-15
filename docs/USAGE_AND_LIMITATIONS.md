# Usage, limitations, and boundaries

I wrote this so nobody mistakes **Anima** for something it isn’t. Read it before you screenshot dashboard numbers for a paper, blog post, or product pitch.

## What this project **is**

- **Instrumentation** for open Hugging Face **causal language models**: hooks on hidden states, scalar **readouts** (valence, arousal, uncertainty-style probe outputs), plus auxiliary signals like entropy, attention summaries, and simple layer-disagreement heuristics.
- A **FastAPI** server (`api/`) and optional **Vite/React** dashboard (`dashboard/`).
- Training helpers (`probes/train.py`, `alignment/`) so you can align probes with **public narrative / atlas-style targets you supply** — the repo doesn’t ship private clinical datasets.

## What this project **is not**

- **Not** proof that a language model **feels**, **has emotions**, or **has subjective experience**.
- **Not** a medical, psychiatric, or therapeutic tool. Don’t use it for diagnosis, treatment, crisis triage, hiring, or law-enforcement scoring.
- **Not** the TRIBE fMRI decoder. The dashboard **TRIBEv2 surrogate** (`alignment/tribe_encoder.py`) is a **deterministic linear sketch** from the **same** activations we probe — useful for demos, **not** voxel-level neural decoding.
- **Not** magically calibrated “emotion AI”: without `probes/zoo/<model_slug>.pt`, the probe weights are **random** — fine for wiring tests, **weak semantics** until you train.

## What you **may** use it for

- **Research and teaching** in interpretability and probing, with careful wording.
- **Prototyping** interfaces that show **internal geometry**, not cartoon feelings.
- **Forking** to extend layer maps (`core/layer_config.py`), train probes, and publish **your** checkpoints under licenses you choose.

## What you **should not** use it for

| Avoid | Why |
|--------|-----|
| Claims like “the model is anxious” stated as **fact** | Numbers are **constructed** from weights and prompts; easy to over-read. |
| **High-stakes** automation | No audited validity, fairness, or safety guarantees. |
| **Mapping humans** onto model readouts | Built for **models**, not people. |
| **Selling** surrogate ROI plots as real neuroscience | Analogies are **labeled metaphors**, not individual brain scans. |

## Models and dependencies

- Each Hugging Face model brings **its own license**. You comply with the Hub terms.
- Gated models need **`HF_TOKEN`** and accepting conditions upstream.

## Privacy

- Don’t paste secrets or regulated personal data through the API unless **you** run a compliant deployment. This codebase isn’t an enterprise security product.

## Accuracy and reliability

- **`hf-internal-testing/tiny-random-gpt2`** is intentionally nonsense text-wise — it’s for CI and low-RAM runs.
- Readouts depend on checkpoint, prompt, decoding settings, and **whether you trained the probe**.
- Windows users sometimes hit PyTorch/paging instability; see [GETTING_STARTED.md](GETTING_STARTED.md).

## Open source compliance

- Anima’s **code** is MIT (see `LICENSE`). Your **data**, **model weights**, and **fork** may add obligations — that’s on you.

## If you cite or demo Anima

Use language like **readout**, **internal state**, **population-level analogy** — see root **README** and [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md). Avoid anthropomorphic certainty.
