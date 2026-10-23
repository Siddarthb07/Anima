# Anima — Claim sheet (IEEE / reviewer facing)

**Source of numbers:** sealed JSON under `benchmarks/reports/` and `benchmarks/reports/ANIMA_NUMBER_TABLE.md`.  
**Do not invent metrics.** If a number is not in a sealed receipt, it is not a claim.

## Thesis (one sentence)

Fixture-perfect trust UIs can **collapse** on live claim-encode evaluation; Anima makes that failure **measurable** (Protocol F→L) and allows a **named** recovery operating point (Protocol L′) without silent label flips or hallucination-SOTA claims.

## Primary claims (supported)

| ID | Claim | Evidence |
|----|--------|----------|
| C1 | On locked HaluEval/TruthfulQA **claim-encode** rows, baseline composite Protocol L **fails** production-style unlock criteria for Qwen and TinyLlama (AUROC, transfer gap, never-abstain, polarity). | `live_guard_*.json`, `trust_integrity_*.json`, Fig. 1 |
| C2 | Fixture Protocol F AUROC ≈ 1.0 is **smoke only** and must not unlock a green trust badge. | GreenShieldBench baseline fail; `honesty.fixture_auroc_cannot_unlock` |
| C3 | Protocol L′ (LOQO ridge on locked rows) is a **separate, named** operating point that can restore AUROC / unlock TIG+GSB without changing production `guard_config.yaml` polarity via silent flips. | `protocol_l_prime_*.json`, `trust_integrity_adapted_*.json`, Fig. 2–3 |
| C4 | Matched semantic-entropy baselines on the same rows do not rescue the baseline composite collapse narrative. | `semantic_entropy_matched_*.json`, Fig. 5 |
| C5 | On a real Narratives laptop subset, **stimulus encoding** (word-rate→BOLD) reaches non-trivial max *r*; Anima PCA→VA holdout does **not** establish brain–emotion alignment. | `fmri_depth_distilgpt2.json`, Fig. 4 |

## Non-claims (explicitly out of scope)

- Hallucination detection SOTA vs commercial or large-model judges.
- That the LM “feels” valence/arousal.
- Voxel-level TRIBEv2 / population fMRI decoding equivalence.
- User-study proof that green badges mislead humans (MGR = **simulated** badge replay).
- Full OpenNeuro ds002345 coverage (depth push = pieman/lucy × 3 subjects).

## Suite roles

| Suite | Role |
|-------|------|
| HaluEval claim-encode | **Primary** unlock suite for adapted criteria |
| TruthfulQA claim-encode | **Secondary** / transfer stress |
| Protocol F fixtures | CI smoke only |
| Council / GoEmotions | Product probe validity (orthogonal to TIG) |

## Operating points

| OP | Meaning | Default product |
|----|---------|-----------------|
| Protocol L baseline | Default composite + live encode | Expected **LOCKED** |
| Protocol L′ adapted | LOQO ridge recovery | **UNLOCKED** only when sealed adapted receipt says so |
