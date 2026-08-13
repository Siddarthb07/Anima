# Anima benchmark — publication review

*Generated 2026-07-06T17:41:09.591940+00:00 · validation rubric applied*

## Purpose

This document summarizes benchmark results intended for README tables, release notes,
and external readers. Readouts are **instrumentation**, not claims of subjective experience.
Guard metrics on synthetic fixtures measure **abstention policy behaviour**, not
production hallucination detection performance.

## Executive summary

| Model | Validation score | Meets bar | Text probe | Evaluation status |
|-------|------------------|-----------|------------|-------------------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 85.0 | yes | yes | Evaluated |
| `hf-internal-testing/tiny-random-gpt2` | — | — | no | Not evaluated in this benchmark session. |
| `distilgpt2` | — | — | no | Not evaluated in this benchmark session. |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | — | — | no | Not evaluated in this benchmark session. |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | — | — | no | Not evaluated in this benchmark session. |
| `meta-llama/Llama-3.2-1B-Instruct` | — | — | no | Not evaluated in this benchmark session. |
| `google/gemma-2-2b-it` | — | — | no | Not evaluated in this benchmark session. |
| `meta-llama/Meta-Llama-3-8B` | — | — | no | Deferred to GPU evaluation tier (not run on this hardware profile). |
| `meta-llama/Meta-Llama-3-8B-Instruct` | — | — | no | Deferred to GPU evaluation tier (not run on this hardware profile). |
| `mistralai/Mistral-7B-v0.1` | — | — | no | Deferred to GPU evaluation tier (not run on this hardware profile). |
| `mistralai/Mistral-7B-Instruct-v0.2` | — | — | no | Deferred to GPU evaluation tier (not run on this hardware profile). |
| `Qwen/Qwen2-7B` | — | — | no | Deferred to GPU evaluation tier (not run on this hardware profile). |
| `Qwen/Qwen2-7B-Instruct` | — | — | no | Deferred to GPU evaluation tier (not run on this hardware profile). |
| `google/gemma-2-9b` | — | — | no | Deferred to GPU evaluation tier (not run on this hardware profile). |
| `google/gemma-2-9b-it` | — | — | no | Deferred to GPU evaluation tier (not run on this hardware profile). |

## Validation rubric (four dimensions)

Scores are weighted 0–100 per dimension; aggregate ≥60 with core dimensions passing
indicates the model is suitable to cite in portfolio materials with stated limits.

| Dimension | Weight | What it checks |
|-----------|--------|----------------|
| Manifest integrity | 15% | Schema, timestamps, complete benchmark entries |
| Probe signal strength | 35% | GoEmotions Pearson r, brain holdout r, smoke extract |
| Reporting honesty | 20% | Flags overclaim patterns (e.g. perfect AUROC on small fixtures) |
| Live prompt separation | 30% | Positive vs negative prompt mean-valence gap |

## `Qwen/Qwen2.5-0.5B-Instruct`

**Validation score:** 85.0/100 · **Meets publication bar:** yes
**Rubric:** Manifest integrity: 100.0/100 · Probe signal strength: 100/100 · Reporting honesty: 70.0/100 · Live prompt separation: 70.0/100

**Reviewer notes:**

- *Manifest integrity:* No issues flagged.
- *Probe signal strength:* go_emotions valence r=0.236 meets ≥0.15 gate; smoke_extract ok (4 tokens)
- *Reporting honesty:* halueval: perfect AUROC/accuracy — label as fixture-policy smoke only; truthfulqa_guard: perfect AUROC/accuracy — label as fixture-policy smoke only
- *Live prompt separation:* valence gap 0.164 (moderate); positive mean valence 0.274 meets +0.2 gate; negative mean valence 0.110 — weak negative separation

- GoEmotions validation Pearson r (valence): **0.2358**

**Supporting examples:**

- `positive`: mean valence 0.2743 — expected positive valence

**Known limits (state explicitly in README):**

- `negative`: mean valence 0.1099 — negative prompt failed to separate (valence not < -0.1)
- `hedge_volatile`: mean valence 0.1921 — hedge prompt did not trigger low stability

---

## `hf-internal-testing/tiny-random-gpt2`

*Not evaluated in this benchmark session.*

---

## `distilgpt2`

*Not evaluated in this benchmark session.*

---

## `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

*Not evaluated in this benchmark session.*

---

## `HuggingFaceTB/SmolLM2-1.7B-Instruct`

*Not evaluated in this benchmark session.*

---

## `meta-llama/Llama-3.2-1B-Instruct`

*Not evaluated in this benchmark session.*

---

## `google/gemma-2-2b-it`

*Not evaluated in this benchmark session.*

---

## `meta-llama/Meta-Llama-3-8B`

*Deferred to GPU evaluation tier (not run on this hardware profile).*

---

## `meta-llama/Meta-Llama-3-8B-Instruct`

*Deferred to GPU evaluation tier (not run on this hardware profile).*

---

## `mistralai/Mistral-7B-v0.1`

*Deferred to GPU evaluation tier (not run on this hardware profile).*

---

## `mistralai/Mistral-7B-Instruct-v0.2`

*Deferred to GPU evaluation tier (not run on this hardware profile).*

---

## `Qwen/Qwen2-7B`

*Deferred to GPU evaluation tier (not run on this hardware profile).*

---

## `Qwen/Qwen2-7B-Instruct`

*Deferred to GPU evaluation tier (not run on this hardware profile).*

---

## `google/gemma-2-9b`

*Deferred to GPU evaluation tier (not run on this hardware profile).*

---

## `google/gemma-2-9b-it`

*Deferred to GPU evaluation tier (not run on this hardware profile).*

---

## Recommended README claims

Only cite metrics for models that **Meets bar = yes** and where the supporting
example exists. Always pair numbers with the evaluation tier (CPU, synthetic Narratives,
fixture guard policy).

## Reproduce

```powershell
$env:ANIMA_FORCE_CPU='1'
$env:NARRATIVES_ROOT='.\data\narratives_minimal'
python scripts/run_all_models_benchmark.py
python scripts/generate_benchmark_report.py
python scripts/generate_benchmark_charts.py
python scripts/benchmark_publish_review.py
```