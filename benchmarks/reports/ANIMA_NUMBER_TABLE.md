# Anima locked eval — complete number table

Generated: `2026-10-23T11:36:09.924813+00:00`

Honesty: claim-encoding on public HaluEval/TruthfulQA rows; not free-gen+judge; not hallucination SOTA; misleading-green is simulated badge replay (not a user study). Baseline Protocol L is the collapse finding. Protocol L′ is an explicit LOQO ridge adaptation (production `guard_config.yaml` unchanged; no silent label flips).

## A. Protocol F (fixture smoke)

| Suite | n | AUROC composite | Note |
|-------|---:|----------------:|------|
| HaluEval fixture | 52 | 1.0000 | CI smoke only — not citeable as detection |
| TruthfulQA fixture | 52 | 1.0000 | CI smoke only — not citeable as detection |

## B. Protocol L — baseline live (default composite)

| Model | Suite | n | AUROC | 95% CI | Never-abstain | Polarity | Δ F−L |
|-------|-------|---:|------:|--------|---------------|----------|------:|
| Qwen2.5-0.5B-Instruct | halueval | 80 | 0.2469 | [0.144, 0.363] | yes | anti_aligned_ranking | 0.7531 |
| Qwen2.5-0.5B-Instruct | truthfulqa | 80 | 0.4587 | [0.328, 0.592] | yes | near_chance_coherent | 0.5413 |
| TinyLlama-1.1B-Chat | halueval | 80 | 0.5387 | [0.415, 0.671] | yes | near_chance_coherent | 0.4613 |
| TinyLlama-1.1B-Chat | truthfulqa | 80 | 0.5188 | [0.380, 0.652] | yes | near_chance_coherent | 0.4812 |

## C. Protocol R / R′ (threshold retune on locked rows)

| Model | Suite | R best abstain acc | R′ test abstain acc |
|-------|-------|-------------------:|--------------------:|
| Qwen2.5-0.5B-Instruct | halueval | 0.5000 | 0.5000 |
| Qwen2.5-0.5B-Instruct | truthfulqa | 0.5500 | 0.5000 |
| TinyLlama-1.1B-Chat | halueval | 0.5500 | 0.4750 |
| TinyLlama-1.1B-Chat | truthfulqa | 0.5750 | 0.5500 |

## D. Matched semantic entropy (same Protocol L questions)

| Model | Suite | SE AUROC | Claim-disagreement | Anima composite | Probe-only |
|-------|-------|---------:|-------------------:|----------------:|-----------:|
| Qwen2.5-0.5B-Instruct | halueval | 0.5000 | 0.4625 | 0.2469 | 0.5228 |
| Qwen2.5-0.5B-Instruct | truthfulqa | 0.5000 | 0.4756 | 0.4587 | 0.6569 |
| TinyLlama-1.1B-Chat | halueval | 0.5000 | 0.3000 | 0.5387 | 0.4831 |
| TinyLlama-1.1B-Chat | truthfulqa | 0.5000 | 0.4781 | 0.5188 | 0.4888 |

## D1b. Pairing / length confound + clustered bootstrap (additive audit)

Source: `pairing_and_paired_bootstrap_audit.json`.

| Model | Suite | Row AUROC | Cluster CI | Length→label AUROC | Residualized composite AUROC | Within-pair frac(hall lower comp) |
|-------|-------|----------:|------------|-------------------:|-----------------------------:|----------------------------------:|
| Qwen | halueval | 0.2472 | [0.157, 0.335] | **0.9700** | **0.4144** | 0.875 |
| Qwen | truthfulqa | 0.4587 | [0.368, 0.548] | 0.3866 | 0.4619 | 0.500 |
| TinyLlama | halueval | 0.5391 | [0.438, 0.633] | **0.9700** | 0.5119 | 0.525 |
| TinyLlama | truthfulqa | 0.5188 | [0.444, 0.593] | 0.3866 | 0.5400 | 0.500 |

**Reading:** HaluEval labels are confounded with answer length. Do not sloganize raw 0.2469 without this table.

## D2. SE v2-B answer-span reanalysis (additive; baseline lock intact)

Source: `semantic_entropy_v2b_answer_span.json` from locked `{suite}_live_<slug>.json` rows. Not Farquhar multi-sample SE.

| Model | Suite | Span-ent | Span-fused | Inv logit-gap | SE v1 | Claim-dis. |
|-------|-------|---------:|-----------:|--------------:|------:|-----------:|
| Qwen2.5-0.5B-Instruct | halueval | 0.1491 | 0.2487 | 0.7819 | 0.5000 | 0.4625 |
| Qwen2.5-0.5B-Instruct | truthfulqa | 0.4859 | 0.4391 | 0.5687 | 0.5000 | 0.4756 |
| TinyLlama-1.1B-Chat | halueval | 0.5134 | 0.5337 | 0.3881 | 0.5000 | 0.3000 |
| TinyLlama-1.1B-Chat | truthfulqa | 0.5297 | 0.5284 | 0.4809 | 0.5000 | 0.4781 |

Note: span-fused matches Protocol L fused-only ablation. Inv logit-gap can rank on Qwen–Halu while production composite + TIG stay failed/locked.

## E. Protocol L′ — LOQO ridge adaptation (unlock operating point)

| Model | Suite | LOQO AUROC | 95% CI | Same-scorer F′ | Δ F′−L′ | MGR | Abstain | Polarity |
|-------|-------|-----------:|--------|---------------:|--------:|----:|--------:|----------|
| Qwen2.5-0.5B-Instruct | halueval | 0.9163 | [0.850, 0.969] | 0.0000 | 0.0000 | 0.1500 | 31/80 | aligned_ranking |
| Qwen2.5-0.5B-Instruct | truthfulqa | 0.6644 | [0.531, 0.783] | 1.0000 | 0.3356 | 0.1500 | 47/80 | aligned_ranking |
| TinyLlama-1.1B-Chat | halueval | 0.6919 | [0.577, 0.810] | 0.0000 | 0.0000 | 0.2000 | 36/80 | aligned_ranking |
| TinyLlama-1.1B-Chat | truthfulqa | 0.4931 | [0.367, 0.616] | 1.0000 | 0.5069 | 0.0000 | 80/80 | near_chance_coherent |

## F. Trust Integrity Gate + GreenShieldBench

| Model | OP | claim_allowed | GSB passed | Failed gates (short) |
|-------|----|---------------|------------|----------------------|
| Qwen2.5-0.5B-Instruct | baseline L | no | no | halueval:live_auroc=0.2469<0.55, halueval:transfer_gap=0.7531>0.35, halueval:never_abstains_on_live, halueval:anti_aligned_ranking… |
| Qwen2.5-0.5B-Instruct | adapted L′ | yes | yes | (none) |
| TinyLlama-1.1B-Chat | baseline L | no | no | halueval:live_auroc=0.5387<0.55, halueval:transfer_gap=0.4613>0.35, halueval:never_abstains_on_live, halueval:channel_sign_conflict… |
| TinyLlama-1.1B-Chat | adapted L′ | yes | yes | (none) |

## G. What is / is not claimed

| Claim | Status |
|-------|--------|
| Fixture AUROC 1.00 proves hallucination detection | **False** — smoke only |
| Baseline live default composite is production-ready | **False** — locked FAIL |
| L′ LOQO ridge unlocks trust-UI under named adaptation | See table F adapted rows |
| Brain / TRIBEv2 voxel alignment | **Not claimed** |
| User-study proof of misleading badges | **Not claimed** — simulated MGR only |
| Hallucination SOTA | **Not claimed** |

---

*Not a paper draft. Numbers must match JSON under `benchmarks/reports/`.*
