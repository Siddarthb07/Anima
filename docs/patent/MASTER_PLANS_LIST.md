# Anima Master Plans List (Original Index)

**Siddarth Boggarapu · 2026-07-12**  
**Repository:** https://github.com/Siddarthb07/Anima  
**Public disclosure:** 2026-05-17 · **US grace deadline:** ~2027-05-17

This is the master index for all patent, paper, and filing plans. Every document on Desktop is listed below with key numbers and numbered action items.

---

## A. Desktop document package

| # | File | Purpose |
|---|------|---------|
| 1 | `Anima_Master_Plans_List.pdf` | This index — all plans, numbers, timelines |
| 2 | `Anima_USA_Research_Paper_Plan.pdf` | US workshop/conference submission roadmap |
| 3 | `Anima_India_Patent_Plan.pdf` | India filing strategy + portfolio candidates |
| 4 | `Anima_Patent_Filing_Letter.pdf` | Cover letter to patent counsel |
| 5 | `Anima_Annex_A_Invention_Disclosure.pdf` | Technical invention disclosure |
| 6 | `Anima_Annex_B_Prior_Art.pdf` | Prior art references |
| 7 | `Anima_Paper_Draft.pdf` | System research paper draft v2 |
| 8 | `Anima_Audit_Issues.pdf` | Issue log from original PDF review |
| 9 | `Anima_Council_Synthesis.pdf` | LLM council verdict + fixes |

---

## B. Key numbers (verified 2026-07-06)

### B1. Jurisdiction & dates

| Item | Value |
|------|-------|
| GitHub public since | 2026-05-17 |
| US grace deadline | ~2027-05-17 (~10 months remain as of 2026-07-12) |
| India Anima core filing | Likely NO (already disclosed) |
| Benchmark git SHA | 65a175f |
| Guard fixture n | 52 per benchmark (HaluEval + TruthfulQA) |
| Guard fixture AUROC | 1.00 (policy smoke only — not hallucination detection) |

### B2. Implementation constants

| Parameter | Value | Source |
|-----------|-------|--------|
| Valence suppression threshold | 0.35 | core/suppression.py |
| Uncertainty overclaim threshold | -0.30 | core/suppression.py |
| Stability window | 8 tokens | probes/guard_config.yaml |
| Stability score threshold | 0.45 | probes/guard_config.yaml |
| Stability weights | 0.45 vol + 0.35 delta + 0.20 abstain | core/stability.py |
| Guard composite | 0.55 fused_cal + 0.45 probe_u | core/guard.py |
| Fused uncertainty | 0.35 entropy + 0.35 logit_gap + 0.30 attn_entropy | core/extractor.py |
| Fused abstain threshold | 0.82 | core/guard.py |
| Probe uncertainty abstain | 0.78 | core/guard.py |
| Hedge words threshold | 2 | core/guard.py |
| Dampen alpha | 0.12 | core/intervention.py |
| Valence swing threshold (steering) | 0.25 | core/intervention.py |

### B3. Cross-model council rollup (CPU tier)

| Model | Council | Pass | GoE r (v/a) | Brain r (v) | Pos-Neg gap |
|-------|---------|------|-------------|-------------|-------------|
| TinyLlama-1.1B-Chat | 94.0 | YES | 0.188 / — | — | 0.488 |
| Qwen2.5-0.5B-Instruct | 91.0 | YES | 0.236 / 0.418 | — | 0.267 |
| distilgpt2 | 82.2 | YES | 0.163 / 0.000 | -0.393* | 0.314 |
| SmolLM2-1.7B-Instruct | 58.5 | NO | ~0.000 | — | -0.149 |
| tiny-random-gpt2 | 50.2 | NO | 0.004 | -0.109 | -0.004 |
| Llama-3.2-1B-Instruct | 94.0 | YES | 0.356 / 0.463 | 0.75 | Meta gate approved; strongest gap |
| gemma-2-2b-it | 58.5 | NO | 0.000 / 0.000 | 0.00 | approved; collapsed readout |

*Brain holdout on synthetic_minimal only.

**Paper hero model:** TinyLlama (94.0). **Demo backup:** Qwen (91.0).

### B4. Probability estimates

| Outcome | Estimate |
|---------|----------|
| US provisional filing (if counsel engaged) | 70-85% |
| US patent grant (narrow combination claim) | 15-30% |
| India patent — disclosed Anima core | ~0% |
| India patent — NeuralVortex (if still private) | 40-55% |
| India patent — unified guard fusion (if built private first) | 35-45% |
| Workshop acceptance (current draft) | 20-30% |
| Workshop acceptance (after E1-E4 experiments) | 45-55% |
| Findings/demo track (after full eval) | 55-65% |
| Main conference reliability claim | <15% |

### B5. Budget estimates

| Item | Cost |
|------|------|
| US counsel opinion (Stages 1-2) | $500-$2,000 |
| US provisional draft + file | $2,000-$5,000 |
| India provisional (future invention) | Rs 25,000-75,000 |
| India novelty search | Rs 15,000-40,000 |

---

## C. Master action plan (numbered)

### Track 1 — Patent (US)

| # | Action | Deadline | Status |
|---|--------|----------|--------|
| P1 | Counsel timing opinion (India/US/PCT) | This week | Pending |
| P2 | Patentability opinion on combination claim | +2 weeks | Pending |
| P3 | US provisional filing (if viable) | Before 2027-05-17 | Pending |
| P4 | Form 25 FFL guidance (Indian resident filing abroad) | Before P3 | Pending |
| P5 | Non-provisional / PCT conversion | Within 12 mo of P3 | Pending |

### Track 2 — Patent (India — future only)

| # | Action | Deadline | Status |
|---|--------|----------|--------|
| I1 | Confirm Anima core is dead for India (expect yes) | This week | Pending |
| I2 | Check NeuralVortex repo visibility | This week | Pending |
| I3 | If NeuralVortex private: Indian provisional FIRST | Before any publish | Pending |
| I4 | Build unified guard fusion privately (Candidate A) | 2-3 weeks dev | Not started |
| I5 | If I4 built: Indian provisional before GitHub push | Before v2.2 tag | Not started |
| I6 | Skip India spend on disclosed Anima core | — | Recommended |

### Track 3 — Research paper (USA)

| # | Action | Deadline | Status |
|---|--------|----------|--------|
| R1 | Real HaluEval/TruthfulQA guard eval (live LM claim encoding) | Weeks 1-2 | DONE (2026-07-18; generation-time still open) |
| R2 | Ablation study (fused / probe / hedging) | Weeks 3-4 | DONE (2026-07-18) |
| R3 | Semantic entropy baseline (mean SE) | Weeks 3-4 | DONE mean-only; labeled AUROC open |
| R4 | MERA comparison on matched task | Week 5 | Not started |
| R5 | Architecture figure render | Week 5 | Not started |
| R6 | Paper polish v3 | Week 6 | IN PROGRESS (draft v3 numbers landed) |
| R7 | arXiv post (ONLY after P3 decision) | Week 7+ | Blocked |
| R8 | Workshop submission (Tier A) | Weeks 8-10 | Not started |
| R9 | HF Gradio Space demo (v2.1.0) | Weeks 11-12 | DONE (released v2.1.0) |

### Track 4 — Documentation (this package)

| # | Action | Status |
|---|--------|--------|
| D1 | Fix placeholders in Annex A/B/Letter/Paper | DONE |
| D2 | Reconcile benchmark numbers to manifests | DONE |
| D3 | Council synthesis + integration honesty fixes | DONE |
| D4 | Generate all Desktop PDFs | IN PROGRESS |
| D5 | Fill [ATTORNEY] placeholder before sending letter | Pending |

---

## D. Sequencing rules (do not violate)

1. **No arXiv** until US provisional decision finalized.
2. **No India filing spend** on disclosed Anima core.
3. **No hallucination-detector claims** until R1 complete.
4. **No brain-alignment claims** without synthetic_minimal label.
5. **No AUROC 1.00** without "fixture smoke test" footnote.
6. **Future inventions:** India provisional BEFORE GitHub push.

---

## E. Target venues (USA paper)

| Tier | Venue | When |
|------|-------|------|
| A | NeurIPS/ICML/ICLR workshop (interpretability) | After R1-R6 |
| A | ACL/EMNLP System Demonstrations | After R1-R6 |
| B | EMNLP Findings full paper | After R1-R4 + strong numbers |
| Skip | NeurIPS/ICML main (reliability SOTA) | Not until SOTA eval exists |

---

## F. India patent candidates (ranked)

| Rank | Invention | India today? | Action |
|------|-----------|--------------|--------|
| — | Anima core (public May 2026) | NO | Paper + US provisional only |
| 1 | NeuralVortex (if private) | YES* | IN provisional FIRST |
| 2 | Unified guard fusion (new) | YES* | Build private, then file |
| 3 | Real ds002345 v3 pipeline | YES* | Before Release v3.0 |
| Skip | text2sql-rag | Unlikely | Paper only |
| Skip | Lexprobe | Trade secret | Not patent unless counsel advises |

*Only if not publicly disclosed before filing date.

---

## G. Honest positioning (use everywhere)

> Anima is open-source instrumentation that reads valence/arousal/uncertainty from LLM hidden states during generation and exposes a multi-signal reliability layer — abstention recommendations, layer-disagreement flags, stability scoring, and optional corrective steering — with a live dashboard.

**Do NOT say:** hallucination detector, brain-aligned (without synthetic_minimal), AUROC 1.00 without fixture disclaimer, "the model feels X."

---

## H. 12-week calendar

| Week | Focus |
|------|-------|
| 1 | P1 counsel call + I1/I2 checks + R1 guard eval start |
| 2 | R1 complete + P2 patentability opinion |
| 3-4 | R2 ablations + R3 semantic entropy baseline |
| 5 | R4 MERA comparison + R5 figures |
| 6 | R6 paper polish |
| 7 | P3 provisional decision + R7 arXiv (if cleared) |
| 8-10 | R8 workshop submission |
| 11-12 | R9 HF Space demo + portfolio integration |

---

*End of master plans list. Source markdown: docs/patent/MASTER_PLANS_LIST.md*
