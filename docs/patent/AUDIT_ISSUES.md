# Anima Patent & Paper — Initial Audit (2026-07-12)

Audit of three source documents:
- `anima-invention-disclosure-annexA.pdf`
- `anima-paper-draft.pdf`
- `anima-patent-filing-letter.pdf`

Cross-checked against live codebase (`github.com/Siddarthb07/Anima`) and benchmark manifests dated 2026-07-06.

---

## Critical (block filing or submission until resolved)

| ID | Document | Issue | Fix applied |
|----|----------|-------|-------------|
| C1 | All | `[repository URL]` placeholder | → `https://github.com/Siddarthb07/Anima` |
| C2 | All | Public disclosure date placeholder | → **2026-05-17** (GitHub repo created public) |
| C3 | Letter, Annex A | India filing likely **barred** — no inventor grace period after public GitHub disclosure | Explicit counsel guidance added; do not imply India filing remains open |
| C4 | Paper | Guard AUROC 1.00 presented without sufficient fixture disclaimer in tables | Reframed as **fixture-policy smoke test**; added manifest `note` field verbatim |
| C5 | Paper | GoEmotions numbers inconsistent (0.21 vs manifest 0.236) | Reconciled to manifest `latest_qwen2.5_0.5b_instruct_manifest.json` |
| C6 | Paper | Missing Annex B (prior art) referenced in letter | Created `ANNEX_B_PRIOR_ART.md` with verified citations |
| C7 | Paper | Related Work entirely TODO | Filled with bibliographically confirmed references |
| C8 | Paper | Abstract lists 3 mechanisms but system has **4** (+ conditional steering) | Corrected to four-mechanism pipeline |
| C9 | Paper | Llama/Gemma marked "not run" but council stubs exist at 48.0 | Clarified: gated models have council stubs only, no full benchmark manifest |

---

## High (patent claim quality / paper credibility)

| ID | Document | Issue | Fix applied |
|----|----------|-------|-------------|
| H1 | Annex A | Prior art table truncated in PDF | Expanded in Annex B with full references |
| H2 | Annex A | Missing auxiliary uncertainty fusion (entropy, logit gap, attn entropy) | Added as supporting (non-claimed) subsystem |
| H3 | Annex A | Missing probe layer-softmax fusion detail | Added in technical description |
| H4 | Annex A | Stability module thresholds not tied to code | Aligned to `probes/guard_config.yaml` and `core/stability.py` |
| H5 | Paper | No architecture diagram | Added ASCII + mermaid figure in paper draft |
| H6 | Paper | No comparison to MERA (Hedstrom et al., ICML 2025) | Added explicit Related Work + differentiation paragraph |
| H7 | Paper | Brain results must stay labeled `synthetic_minimal` | Enforced throughout |
| H8 | Letter | Contact placeholders | Filled with inventor details from portfolio record |
| H9 | Patent | Combination claim must distinguish from MERA abstention-only path | Added differentiation in Annex A §5 |
| H10 | Paper | Missing fused-uncertainty formula | Added from `core/extractor.py` |

---

## Medium (polish / sequencing)

| ID | Document | Issue | Fix applied |
|----|----------|-------|-------------|
| M1 | Letter | Attorney placeholders remain (user must fill) | Marked `[ATTORNEY — fill before send]` |
| M2 | Paper | Affiliation TBD | Set to independent / VIBGYOR + GitHub org |
| M3 | Annex A §6 | Commercial context empty | Added open-source + optional API integration note |
| M4 | Paper | Section 7 ablations still "not yet run" | Kept honest; added priority ordering |
| M5 | Paper | Sequencing: arXiv after provisional | Preserved in checklist |
| M6 | All | Inconsistent mechanism numbering (3.4 vs 3.5 steering) | Renumbered consistently |

---

## Verified constants (code ↔ documents)

| Parameter | Value | Source |
|-----------|-------|--------|
| `VALENCE_SUPPRESSION_THRESHOLD` | 0.35 | `core/suppression.py` |
| `UNCERTAINTY_OVERCLAIM_THRESHOLD` | −0.30 | `core/suppression.py` |
| Stability window | 8 tokens | `probes/guard_config.yaml` |
| Stability score threshold | 0.45 | `probes/guard_config.yaml` |
| Stability weights | 0.45 vol + 0.35 delta + 0.20 abstain | `core/stability.py` |
| Guard composite | 0.55 fused_cal + 0.45 probe_u | `core/guard.py` |
| Fused uncertainty | 0.35 entropy + 0.35 logit_gap + 0.30 attn_entropy | `core/extractor.py` |
| Dampen alpha | 0.12 | `core/intervention.py` |
| Dampen valence delta threshold | 0.25 | `core/intervention.py` |

---

## Jurisdiction timing summary (for counsel — not legal advice)

| Jurisdiction | Status as of 2026-07-12 | Notes |
|--------------|-------------------------|-------|
| **India** | Likely **not available** for this disclosure | Public GitHub since 2026-05-17; Section 31/39 Indian Patents Act — no general inventor grace period |
| **United States** | **Likely available** under 35 U.S.C. § 102(b)(1)(A) | 12-month grace from 2026-05-17 → deadline **~2027-05-17** (~10 months remain) |
| **PCT** | Depends on US/residency strategy | Must file before grace expires; India as designated state may be moot if Indian rights lost |

---

## Benchmark numbers (authoritative: 2026-07-06 manifests)

| Model | Council | GoE r (v/a) | Brain r (v) | Pos−Neg gap |
|-------|---------|-------------|-------------|-------------|
| Qwen2.5-0.5B-Instruct | 91.0 ✓ | 0.236 / 0.418 | — | 0.267 |
| TinyLlama-1.1B-Chat | 94.0 ✓ | 0.188 / — | — | 0.488 |
| distilgpt2 | 82.2 ✓ | 0.163 / 0.000 | −0.393 | 0.314 |
| SmolLM2-1.7B-Instruct | 58.5 ✗ | ~0.000 | — | −0.149 (inverted) |
| tiny-random-gpt2 | 50.2 ✗ | 0.004 | −0.109 | −0.004 |
| Llama-3.2-1B-Instruct | 94.0 ✓ | GoE r=0.36/0.46; gap 0.75 | strong passer | Meta gate approved |
| gemma-2-2b-it | 58.5 ✗ | GoE r=0; gap 0.00 | collapsed valence=1.0 | approved; fail honestly |

Guard fixtures: n=52, AUROC=1.00 — **policy smoke test on pre-labeled readout rows**, not end-to-end LM hallucination detection.

---

## Council Round 2 (2026-07-12) — integration honesty fixes

| ID | Issue | Fix |
|----|-------|-----|
| F1 | Four mechanisms described as one fused abstention pipeline — **code shows parallel paths** | Reframed in Annex A, Paper §3.5, Letter §3; see `COUNCIL_SYNTHESIS.md` |
| F2 | Validation steering uses uncertainty head; production steering uses valence head | Clarified in Annex A §3.5 and Paper §3.6/3.7 |
| F3 | Default API modes (`observe`/`none`) weaken implicit "always gating" claim | Documented in Annex A table, Paper §4 |
| F4 | MERA prior art strength underweighted in letter | Added explicit counsel warning |

**Full council report:** `docs/patent/COUNCIL_SYNTHESIS.md`
