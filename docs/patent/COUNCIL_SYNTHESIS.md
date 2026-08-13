# LLM Council Synthesis — Anima Patent & Paper Package

**Date:** 2026-07-12  
**Status:** Complete — council subagents timed out on first run; synthesis + Round 2 fixes applied inline (resumed 2026-07-12)  
**Documents reviewed:** `docs/patent/*`, `core/guard.py`, `core/stability.py`, `core/suppression.py`, `core/intervention.py`, `api/server.py`

---

## Executive verdict

| Question | Verdict |
|----------|---------|
| **US provisional filing** | **Conditional YES** — file only if counsel confirms 12-month grace (deadline ~2027-05-17) and drafts **narrow combination claims** |
| **India filing** | **Likely NO** — public GitHub since 2026-05-17 |
| **Paper submission now** | **NO** for hallucination-detection framing; **YES** for systems/tooling workshop after minor fixes |
| **Patentability confidence** | **Low–medium** — MERA (ICML 2025) is strong combination prior art; affect framing alone is weak differentiation |

---

## Council-devils-advocate — Fatal & high issues

### F1 (FATAL for broad claims): Mechanisms are **parallel, not fully fused**

**Finding:** Layer-disagreement (`detect_suppression`) and rolling stability (`analyze_readout_stability`) do **not** feed into `evaluate_guard()` abstention logic. Guard abstains on:

- `fused_calibrated ≥ 0.82`
- `probe_uncertainty ≥ 0.78`
- `lexical_hedging ≥ 2` words
- Rule: ≥2 reasons OR fused alone

Stability **only** relabels region to `high-uncertainty` when `guard_mode="gate"` (default is `"observe"`). Suppression events are returned separately in API response — they do **not** trigger abstain.

**Fix required:** Reframe "one gating pipeline" as **"integrated reliability monitoring system with coordinated outputs"** — not a single fused abstention score over all four signals. Patent claims should describe **system combination** (hooks → probes → parallel evaluators → optional interventions), not imply all four signals vote on one abstain bit.

### F2 (FATAL for India): Public disclosure already occurred

Already documented. Do not spend counsel budget assuming Indian rights remain.

### F3 (HIGH): MERA overlap

Hedstrom et al. (ICML 2025) already combines **abstention + calibrated steering**. Anima's differentiation must emphasize:
- No helper model
- Layer-disagreement as separate surfaced signal
- Rolling stability gate (opt-in `guard_mode`)
- Affect-probe training target (GoEmotions circumplex)

This may not survive obviousness rejection without dependent claims on specific thresholds.

### F4 (HIGH): "Affect probe" ≠ novel probe type

Valence/arousal/uncertainty heads are still linear probes on hidden states. The **training target** differs (Russell circumplex vs truthfulness), not the mechanism. Patent should claim **combination + affect-framed readout space**, not "novel probe."

### F5 (HIGH): Guard AUROC 1.00

Fixtures are **synthetic readout rows** with perfect separability by design. Any patent example or paper table implying hallucination detection is **misleading**. Already partially fixed; must stay in every results table footnote.

### F6 (MEDIUM): Steering direction inconsistency

- **Causal validation** (`probes/validate.py`): steers along **uncertainty** head weights
- **Production intervention** (`core/intervention.py`): steers along **valence** head weights

Paper §3.5 and §3.6 must distinguish these explicitly.

### F7 (MEDIUM): Default API modes weaken "real-time gating" claim

Defaults: `guard_mode="observe"`, `intervention_mode="none"`. Full four-mechanism behavior requires explicit API flags. Disclosure must state this.

---

## Council-architect — Architecture fixes

### A1. Add signal-flow table (implemented in Annex A v2)

| Mechanism | Triggers abstain? | Default active? | Output surface |
|-----------|-------------------|-----------------|----------------|
| Guard fusion | **Yes** (`abstain_recommended`) | Always | Per-token `guard` field |
| Layer disagreement | No | Yes (`detect_suppression=true`) | `suppression_events[]` |
| Rolling stability | No (relables region only) | Analysis always; gate opt-in | `flags.unstable_window`, summary |
| Conditional steering | N/A | Opt-in (`intervention_mode=dampen`) | Modified generation |

### A2. Recommended patent claim structure

**Independent claim 1 (system):** A computer-implemented system comprising: (a) forward hooks on early and late transformer layers; (b) affect-framed linear probe heads outputting valence, arousal, uncertainty; (c) layer-disagreement detector comparing early/late readouts; (d) rolling-window stability scorer; (e) multi-signal abstention evaluator combining calibrated fused uncertainty, probe uncertainty, and lexical hedging; (f) conditional residual-stream steering module triggered by valence swing threshold; wherein (c)–(f) operate during autoregressive generation on a common per-token readout stream.

**Dependent claims:** specific thresholds (0.35, −0.30, 0.45, 0.82, 0.25, 0.12); window size 8; fusion weights; Platt calibration; random-direction control validation.

**Do not claim:** TRIBEv2 surrogate, dashboard UI, GoEmotions training per se.

### A3. Missing from original PDFs (now added)

- Auxiliary uncertainty fusion formula
- Layer softmax fusion in probe
- API mode defaults (`observe` / `none`)
- Signal integration honesty (F1)

---

## Council-pragmatist — Priority action list

| # | Action | Timeline | Blocks |
|---|--------|----------|--------|
| 1 | **Counsel timing call** (US grace only) | This week | Everything |
| 2 | **US provisional** (if counsel agrees) | Before 2027-05-17 | arXiv |
| 3 | **Fix integration language** in all docs (F1) | Done in this pass | Patent draft |
| 4 | **Real HaluEval guard eval** (live LM generations) | 2–3 weeks | Paper reliability claim |
| 5 | **Ablation: guard signals disabled individually** | 1 week after #4 | Paper |
| 6 | **MERA comparison** on matched task | 2 weeks | Paper Related Work |
| 7 | **Semantic entropy baseline** | 1 week | Paper |
| 8 | **Workshop submission** (interpretability track) | After #4–#6 | — |
| 9 | **Skip India filing spend** unless counsel finds exception | — | Budget |
| 10 | **Do NOT** claim SOTA hallucination detection before #4–#7 | — | Credibility |

**Minimum for provisional:** Annex A + B + letter (done). No new experiments required for provisional filing.

**Minimum for paper:** #4 + #5 + architecture figure render.

---

## Council-product — Positioning fixes

### Honest one-liner
> "Anima is open-source instrumentation that reads valence/arousal/uncertainty from LLM hidden states during generation and exposes a multi-signal reliability layer — abstention recommendations, layer-disagreement flags, stability scoring, and optional corrective steering — with a live dashboard."

### Do NOT say
- "Hallucination detector" (until real benchmark)
- "The model feels X"
- "Brain-aligned" without `synthetic_minimal` label
- "AUROC 1.00" without "fixture smoke test"

### Affect framing
**Helps paper** (differentiates from ITI/MERA truthfulness framing). **Weak for patent** (examiner will say it's just a different probe label). Use affect framing in paper; use **system combination** in patent claims.

---

## Fixes applied in this pass (Council round 2)

| ID | File | Change |
|----|------|--------|
| R2-1 | ANNEX_A | Added §3.0 signal integration table; clarified parallel vs fused paths |
| R2-2 | ANNEX_A | Added API default modes note |
| R2-3 | PAPER_DRAFT | Fixed steering vs validation direction distinction |
| R2-4 | PAPER_DRAFT | Added integration honesty paragraph in §3 |
| R2-5 | PAPER_DRAFT | Added `guard_mode` / `intervention_mode` to experimental setup |
| R2-6 | PATENT_FILING_LETTER | Added counsel warning on MERA prior art strength |
| R2-7 | AUDIT_ISSUES | Added council round 2 findings |

---

## Sign-off

**Proceed with US provisional:** Yes, if budget allows and counsel confirms grace window — but expect **narrow claims** and **low prosecution confidence**.

**Proceed with paper (workshop):** Yes, after real guard eval — current draft is honest enough for a **systems/tooling** submission, not a reliability SOTA claim.

**Proceed with India filing:** No, unless counsel identifies a viable exception (unlikely).
