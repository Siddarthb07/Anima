# Annex B — Prior Art References

**Invention:** Anima — Multi-Signal Reliability Gating System for Language Model Generation  
**Prepared by:** Siddarth Boggarapu  
**Date:** 2026-07-12  
**Status:** Preliminary inventor search — **counsel must conduct independent search**

---

## A. Linear probing / hidden-state signals for hallucination and uncertainty

### A1. Marks & Tegmark (2023)
**Title:** The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Statements  
**Venue:** arXiv:2310.06824  
**Relevance:** Identifies linear directions in activation space associated with truthfulness; foundational for probe-based reliability signals.  
**Differentiation:** Anima uses affect-style (valence/arousal/uncertainty) probes, not binary true/false classification.

### A2. Thai et al. (2025) — Factuality Probes
**Title:** Simple Factuality Probes Detect Hallucinations in Long-Form Natural Language Generation  
**Venue:** Findings of EMNLP 2025 (DOI: 10.18653/v1/2025.findings-emnlp.880)  
**Relevance:** Linear probes on hidden states for claim-level hallucination detection; single-signal, offline evaluation.  
**Differentiation:** Anima integrates layer disagreement + stability + lexical hedging into a real-time gating pipeline; not claim-level factuality probes.

### A3. O'Neill et al. (2025)
**Title:** A Single Direction of Truth: An Observer Model's Linear Residual Probe Exposes and Steers Contextual Hallucinations  
**Venue:** arXiv:2507.23221  
**Relevance:** Single linear direction for hallucination detection + causal steering; observer-model architecture.  
**Differentiation:** Anima uses multi-mechanism gating (not single direction) and affect framing.

### A4. Obeso & Arditi et al. (2025) — Hallucination Probes
**Title:** Real-Time Detection of Hallucinated Entities in Long-Form Generation  
**URL:** https://www.hallucination-probes.com/  
**Relevance:** Token-level linear probes for entity hallucination during streaming generation.  
**Differentiation:** Entity-level token probes vs Anima's affect + layer-disagreement + stability fusion.

### A5. Li et al. (2023) — Semantic Entropy (baseline, not probe)
**Title:** Generating with Confidence: Uncertainty Quantification for Black-box LLMs  
**Venue:** UAI 2023 / arXiv:2305.19187  
**Relevance:** Sampling-based uncertainty (semantic entropy) — standard UQ baseline Anima has not yet benchmarked against.  
**Differentiation:** Output/sampling-based vs white-box internal-state gating.

---

## B. Activation steering

### B1. Li et al. (2023/2024) — Inference-Time Intervention (ITI)
**Title:** Inference-Time Intervention: Eliciting Truthful Answers from a Language Model  
**Venue:** NeurIPS 2023 (arXiv:2306.03341)  
**Relevance:** Shifts activations along learned truthful directions at inference; linear probing on attention heads.  
**Differentiation:** ITI applies **unconditional** steering toward truthfulness; Anima applies **conditional** steering only on detected valence swings within a multi-signal gating pipeline.

### B2. Rimsky et al. (2024) — Contrastive Activation Addition (CAA)
**Title:** Steering Llama 2 via Contrastive Activation Addition  
**Venue:** ACL 2024 (DOI: 10.18653/v1/2024.acl-long.828)  
**Relevance:** Steering vectors from contrastive activation differences; added at all token positions.  
**Differentiation:** CAA is unconditional, contrast-pair derived; Anima steering is probe-weight derived and gated.

### B3. Zou et al. (2023) — Representation Engineering (RepE)
**Title:** Representation Engineering: A Top-Down Approach to AI Transparency  
**Venue:** arXiv:2310.01405  
**Relevance:** Population-level representation monitoring and manipulation for honesty, harmlessness, etc.  
**Differentiation:** RepE is a general transparency framework; Anima is a specific four-mechanism reliability pipeline with affect probes.

---

## C. Layer-wise internal signals

### C1. Multiple 2024–2026 works
**Examples:** Layer-wise information flow analysis; INSIDE (EigenScore); mid-layer linear decodability of truthfulness (Aiersilan 2026, arXiv:2606.02628).  
**Relevance:** Early/middle/late layer comparisons for hallucination signal.  
**Differentiation:** Anima compares **the same affect probe head** applied to early vs late layers within one forward pass, feeding a gating policy — not standalone layer analysis.

---

## D. Abstention + steering (closest combination prior art)

### D1. Hedstrom et al. (2025) — MERA
**Title:** To Steer or Not to Steer? Mechanistic Error Reduction with Abstention for Language Models  
**Venue:** ICML 2025, PMLR 267:22924-22945 (arXiv:2510.13290)  
**Relevance:** **Closest prior art to Anima's guard design.** Combines calibrated abstention with adaptive activation steering via a helper model.  
**Differentiation:**
- MERA uses a **separate helper model** for error prediction; Anima uses **probe-derived affect signals** directly.
- MERA does not include **layer-disagreement detection** or **rolling temporal stability scoring**.
- Anima's probe framing is **valence/arousal/uncertainty**, not truthfulness/error classification.
- Anima adds **lexical hedging** as an independent abstention reason fused with probe signals.

### D2. Wang et al. (2025) — Seal
**Title:** Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning  
**Venue:** ACL 2025 (DOI: 10.18653/v1/2025.acl-long.1199)  
**Relevance:** Training-time abstention via [REJ] token; different lifecycle (fine-tuning vs inference-time gating).

---

## E. Affect / emotion probing in LLMs

### E1. GoEmotions (Demszky et al., 2020)
**Title:** GoEmotions: A Dataset of Fine-Grained Emotions  
**Venue:** ACL 2020  
**Relevance:** Text-emotion labels mapped to Russell circumplex valence/arousal targets for probe training.  
**Differentiation:** GoEmotions is a dataset; Anima's contribution is wiring affect probes into a reliability gating pipeline.

---

## F. Control methodology (not independently novel)

### F1. Random-direction steering controls
**Relevance:** Standard validation in CAA, ITI follow-ups, and Anima's `probes/validate.py`.  
**Status:** Methodology only; not part of claimed combination.

---

## G. Suggested claim-drafting notes for counsel

1. **Draft narrowly** around the four-mechanism combination + affect-probe framing + real-time pipeline.
2. **Avoid claiming** individual components (linear probing, steering, abstention) standalone.
3. **Include dependent claims** on specific threshold ranges and fusion weights as implemented.
4. **Do not overclaim** guard performance — fixture validation only; no labeled hallucination benchmark yet.
5. **MERA (D1)** is the primary combination prior art to distinguish against in prosecution.

---

## Bibliography (BibTeX-ready)

```bibtex
@inproceedings{li2023iti,
  title={Inference-Time Intervention: Eliciting Truthful Answers from a Language Model},
  author={Li, Kenneth and Patel, Oam and Vi{\'e}gas, Fernanda and Pfister, Hanspeter and Wattenberg, Martin},
  booktitle={NeurIPS},
  year={2023}
}

@inproceedings{rimsky2024caa,
  title={Steering Llama 2 via Contrastive Activation Addition},
  author={Rimsky, Nina and Gabrieli, Nick and Schulz, Julian and Tong, Meg and Hubinger, Evan and Turner, Alexander},
  booktitle={ACL},
  year={2024}
}

@misc{zou2023repe,
  title={Representation Engineering: A Top-Down Approach to AI Transparency},
  author={Zou, Andy and others},
  year={2023},
  eprint={2310.01405},
  archivePrefix={arXiv}
}

@inproceedings{thai2025factuality,
  title={Simple Factuality Probes Detect Hallucinations in Long-Form Natural Language Generation},
  author={Thai, Jason and others},
  booktitle={Findings of EMNLP},
  year={2025}
}

@inproceedings{hedstrom2025mera,
  title={To Steer or Not to Steer? Mechanistic Error Reduction with Abstention for Language Models},
  author={Hedstrom, Andreas and others},
  booktitle={ICML},
  year={2025}
}
```
