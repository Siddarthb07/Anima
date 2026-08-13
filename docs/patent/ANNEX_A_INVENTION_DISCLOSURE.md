# Annex A — Invention Disclosure

**Invention:** Multi-Signal Reliability Gating System for Language Model Generation (“Anima”)  
**Inventor:** Siddarth Boggarapu  
**Date:** 2026-07-12  
**Repository:** https://github.com/Siddarthb07/Anima (MIT license)

---

## 1. Background and Disclosure Timing

The codebase implementing this invention is public on GitHub at **https://github.com/Siddarthb07/Anima**, created as a **public** repository on **2026-05-17** (UTC). This constitutes a public disclosure by the inventor.

**Counsel action required:** Confirm jurisdictional consequences before any filing. Under Indian patent law, inventor self-disclosure before filing generally destroys novelty with no grace period. Under U.S. law (35 U.S.C. § 102(b)(1)(A)), a provisional application may remain viable if filed within **12 months** of the 2026-05-17 disclosure date (deadline approximately **2027-05-17**).

No academic paper or preprint describing this system has been published as of the date of this disclosure. A draft system paper is attached for sequencing reference only.

---

## 2. Field and Problem

The invention relates to **reliability monitoring for large language model (LLM) text generation** — specifically, detecting during autoregressive generation (before output is returned) when an LLM's output may be unreliable (e.g., hallucinated or overconfident), by monitoring **internal hidden-state activations** rather than analyzing finished output text alone.

Existing approaches typically validate a **single** internal signal (one probe type, one steering direction) offline. Anima integrates multiple real-time evaluators on a shared per-token readout stream: a guard module that can **recommend abstention**, parallel **layer-disagreement** and **stability** signals, and an optional **conditional corrective steering** step.

---

## 3. Description of the System

### 3.0 Overview architecture

```
Prompt → HF causal LM (forward hooks on N layers)
       → AffectProbe (layer-softmax fusion → valence, arousal, uncertainty)
       → Auxiliary uncertainty (entropy + logit gap + attention entropy → fused)
       → Four-mechanism gating pipeline:
            (i)   layer-disagreement detection
            (ii)  rolling temporal stability scoring
            (iii) calibrated multi-signal fusion → abstention
            (iv)  conditional corrective steering
       → REST/WebSocket API + optional dashboard
```

The system attaches lightweight **probe neural networks** to specific transformer layers via **forward hooks**. At each generated token, hidden states from multiple layers are read out and passed through probe heads outputting three scalars per token:

| Output | Range | Role |
|--------|-------|------|
| Valence | [−1, +1] | Pleasant ↔ unpleasant (Russell circumplex framing) |
| Arousal | [0, 1] | Activated ↔ calm |
| Uncertainty | [0, 1] | Epistemic spread (probe head + auxiliary signals) |

**Layer fusion (probe readout):** Activations from N hooked layers are combined via learned softmax weights before the three linear heads are applied. The same head weights are also applied per-layer via `heads_from_hidden()` for early/late comparison.

**Auxiliary uncertainty (supporting, not individually claimed):** Independent of the probe uncertainty head, three logits/attention signals are fused:

```
fused = 0.35·entropy + 0.35·logit_gap + 0.30·attn_entropy
```

Optional **Platt scaling** calibrates fused scores on held-out labeled pairs.

Four mechanisms operate on the shared per-token readout stream during generation. **The combination as an integrated real-time reliability system** — not any single mechanism in isolation — is the subject of the novelty inquiry, applied through an **affect-style (valence/arousal/uncertainty) probe framing** rather than a raw truthfulness classifier.

**Signal integration (important for claim drafting):** The four mechanisms are **coordinated but not all fused into one abstention score**:

| Mechanism | Drives `abstain_recommended`? | Default API mode |
|-----------|------------------------------|------------------|
| Guard fusion (§3.3) | **Yes** | Always active |
| Layer disagreement (§3.1) | No — surfaced as `suppression_events` | `detect_suppression=true` |
| Rolling stability (§3.2) | No — relabels region when `guard_mode="gate"` | Default `guard_mode="observe"` |
| Conditional steering (§3.4) | N/A — modifies next forward step | Default `intervention_mode="none"` |

Counsel should draft claims around the **system combination** (shared hook/probe infrastructure + four coordinated evaluators/interventions), not imply a single fused abstention function over all four signals.

---

### 3.1 Layer-disagreement detection

The same probe head is applied to hidden states from an **early layer** and a **late layer** of the same forward pass:

```
valence_shift     = late_affect.valence − early_affect.valence
uncertainty_shift = late_affect.uncertainty − early_affect.uncertainty
```

| Flag | Condition | Interpretation |
|------|-----------|----------------|
| `valence_suppression` | valence_shift > **0.35** | Late layers show more positive valence than early layers |
| `uncertainty_overclaim` | uncertainty_shift < **−0.30** | Late layers under-report uncertainty vs early layers |

These are heuristic internal-inconsistency signals, not claims about subjective experience or deception.

**Implementation:** `core/suppression.py`

---

### 3.2 Rolling temporal stability scoring

Over a sliding window of **W = 8** recent tokens (configurable):

```
stability_score = max(0, 1 − (0.45·vol_penalty + 0.35·delta_penalty + 0.20·abstain_penalty))
```

Where:
- `vol_penalty` = normalized valence volatility within the window
- `delta_penalty` = mean token-to-token valence change
- `abstain_penalty` = recent guard abstention rate, normalized by threshold 0.5

If `stability_score < 0.45` (default), the window is flagged **unstable** and, in gate mode, forces downstream region labeling to **high-uncertainty**.

**Implementation:** `core/stability.py`, thresholds in `probes/guard_config.yaml`

---

### 3.3 Calibrated multi-signal fusion for abstention

A composite score combines calibrated fused uncertainty with probe-level uncertainty:

```
composite = 0.55·fused_calibrated + 0.45·probe_uncertainty
```

**Abstain** when:
```
(count(independent_reasons) ≥ 2) OR (fused_calibrated ≥ 0.82)
```

Independent reasons include:
1. `high_fused_uncertainty` — fused_calibrated ≥ 0.82
2. `high_probe_uncertainty` — probe uncertainty ≥ 0.78
3. `lexical_hedging` — hedge-word count in generated token text ≥ 2 (e.g., "maybe", "perhaps", "I think")

When abstaining, region labels are overridden to `high-uncertainty`.

**Implementation:** `core/guard.py`

**Important evaluation caveat:** Abstention logic has been validated against **n=52 synthetic fixture rows** with pre-labeled readout fields (AUROC 1.00). This confirms the **policy executes correctly** on supplied inputs; it does **not** demonstrate end-to-end hallucination detection on live LM generations at scale.

---

### 3.4 Conditional corrective steering

When recent valence history shows a swing ≥ **0.25** between consecutive tokens, a **one-step** correction is applied to the model's internal activations:

1. Extract valence direction from probe head weights: `v_w / ||v_w||`
2. Register a temporary forward hook on a mid-layer residual stream
3. Add `±α · direction` (default α = **0.12**) opposite the swing sign
4. Remove hook after one forward step

This dampens large valence swings before the next token is generated. Steering is **conditional** on detected instability, unlike unconditional ITI/CAA-style intervention.

**Implementation:** `core/intervention.py`

---

### 3.5 Causal validation of probe directions (supporting method, not claimed as novel)

To verify a probe-derived direction has causal effect (not merely correlational), the diagnostic harness (`probes/validate.py`):
1. Steers along the probe's **uncertainty-head** weight direction
2. Compares hedge-word rate against steering along **n random unit directions** of equal norm
3. Reports `signal_is_real = experimental_score > max(control_scores)`

**Note:** Production corrective steering (§3.4) uses the **valence-head** direction, not the uncertainty head. Validation and intervention intentionally probe different axes.

**Implementation:** Standard control methodology in activation-steering literature; included for completeness, not part of the claimed combination.

---

## 4. Prior Art Identified to Date

See **Annex B** for full bibliographic references. Summary:

| Component | Closest prior art |
|-----------|-------------------|
| Linear probing of hidden states for uncertainty/hallucination | Marks & Tegmark 2023; Thai et al. (Factuality Probes) EMNLP 2025 Findings; O'Neill et al. 2025 |
| Activation steering along learned directions | Li et al. (ITI) NeurIPS 2023; Rimsky et al. (CAA) ACL 2024; Zou et al. (RepE) arXiv 2023 |
| Random-direction control validation for steering | Standard in steering literature (e.g., CAA ablations) |
| Layer-wise divergence as hallucination signal | Multiple 2024–2026 papers on early/middle/late layer comparison |
| Multi-signal fusion driving abstention | Hedstrom et al. (MERA) ICML 2025 — abstention + calibrated steering |
| Affect-style (valence/arousal) probe framing | GoEmotions/Russell circumplex mapping in NLP; not previously combined with the four-mechanism gating pipeline above |

---

## 5. Inventor's Preliminary View on Novelty (for Counsel to Test, Not Adopt)

No individual mechanism in Section 3.1–3.4 is believed novel in isolation. The inventor's tentative view is that the **specific combination** of:

1. layer-disagreement detection (3.1),
2. rolling stability scoring (3.2),
3. calibrated multi-signal fusion abstention (3.3), and
4. conditional corrective steering (3.4),

operating together as **one real-time gating pipeline** and applied through an **affect-style (valence/arousal/uncertainty) probe framing** rather than a truthfulness-style classifier, may be narrow enough to be non-obvious over any single piece of identified prior art — particularly MERA (Hedstrom et al., 2025), which addresses abstention + steering but not layer-disagreement or temporal stability scoring in an affect-probe framing.

**Differentiation from MERA:** MERA uses a helper model to predict error likelihood and calibrate steering strength; Anima uses probe-derived affect signals + layer disagreement + rolling stability + lexical hedging fusion without a separate helper model, and applies steering only on detected valence swings.

This is explicitly offered as a **hypothesis for counsel's independent evaluation**.

---

## 6. Commercial Context

- **Current release:** Open-source (MIT) with live React dashboard and FastAPI backend.
- **No proprietary product deployment** as of disclosure date.
- **Potential paths (for claim-strategy discussion):** API middleware for RAG pipelines (e.g., legal/medical document generation), Hugging Face inference plugin, enterprise guard-rail layer. None committed.

---

## 7. Materials Available on Request

- Full source code: https://github.com/Siddarthb07/Anima
- Per-module technical walkthrough: `docs/ANIMA_TECHNICAL_OVERVIEW.md`
- Preliminary benchmark manifests: `benchmarks/reports/` (2026-07-06 CPU-tier rollup)
- Draft academic paper (attached separately; not published)
- Benchmark council reports: `benchmarks/reports/council_rollup.json`

**Evaluation gap (relevant to claim breadth):** Guard/abstention validated on synthetic fixtures only; real HaluEval/TruthfulQA end-to-end runs not yet completed. Probe training validated on GoEmotions (text) and synthetic-minimal Narratives (brain-aligned placeholder).
