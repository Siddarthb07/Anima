# Anima benchmark report

*Generated 2026-07-06T17:41:09.591940+00:00 · validation rubric applied*

## Executive summary

This report aggregates benchmarks across all registered HuggingFace models in `core/layer_config.py`.
Readouts are **instrumentation**, not claims that models feel emotions.
Guard AUROC scores on synthetic fixtures are **policy smoke tests**, not hallucination detection benchmarks.

| Model | Validation score | Meets bar | Text probe | Brain probe |
|-------|------------------|-----------|------------|-------------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 85.0 | yes | yes | no |
| `hf-internal-testing/tiny-random-gpt2` | — | — | no | no |
| `distilgpt2` | — | — | no | no |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | — | — | no | no |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | — | — | no | no |
| `EleutherAI/pythia-410m` | — | — | no | no |
| `facebook/opt-350m` | — | — | no | no |
| `meta-llama/Llama-3.2-1B-Instruct` | — | — | no | no |
| `google/gemma-2-2b-it` | — | — | no | no |
| `meta-llama/Meta-Llama-3-8B` | — | — | no | no |
| `meta-llama/Meta-Llama-3-8B-Instruct` | — | — | no | no |
| `mistralai/Mistral-7B-v0.1` | — | — | no | no |
| `mistralai/Mistral-7B-Instruct-v0.2` | — | — | no | no |
| `Qwen/Qwen2-7B` | — | — | no | no |
| `Qwen/Qwen2-7B-Instruct` | — | — | no | no |
| `google/gemma-2-9b` | — | — | no | no |
| `google/gemma-2-9b-it` | — | — | no | no |

## Benchmark validation rubric

Four rule-based dimensions score each model (0–100, weighted):

1. **schema_integrity** (15%) — manifest schema, timestamps, entries present
2. **probe_signal** (35%) — GoEmotions Pearson r, brain holdout r, smoke extract
3. **honesty_flags** (20%) — penalises perfect AUROC on tiny fixtures, n<50
4. **prompt_separation** (30%) — positive vs negative live prompt mean-valence gap

Aggregate ≥60 with core dimensions passing = **meets publication bar**.

## `Qwen/Qwen2.5-0.5B-Instruct`

**Validation score:** 85.0/100 · **Meets bar:** yes

### Benchmark entries

| Benchmark | Status | Metrics |
|-----------|--------|---------|
| smoke_extract | ok | n=4 |
| narratives_dev | skipped | Use run_narratives_encoding.py for holdout metrics; dev runner reserved for CI subset |
| go_emotions | ok | r_v=0.2358 |
| halueval | ok | guard_acc=1.0 |
| truthfulqa_guard | ok | guard_acc=1.0 |

### Rubric dimension notes

- **schema_integrity** (100.0/100): 
- **probe_signal** (100/100): go_emotions valence r=0.236 meets ≥0.15 gate; smoke_extract ok (4 tokens)
- **honesty_flags** (70.0/100): halueval: perfect AUROC/accuracy — label as fixture-policy smoke only; truthfulqa_guard: perfect AUROC/accuracy — label as fixture-policy smoke only
- **prompt_separation** (70.0/100): valence gap 0.164 (moderate); positive mean valence 0.274 meets +0.2 gate; negative mean valence 0.110 — weak negative separation

### Good examples (thesis-supporting)

**Good — `positive`**
- Prompt: *I'm thrilled — we finally shipped it and everything worked.*
- Mean valence: **0.2743** · arousal: 0.5586
- Rubric note: expected positive valence
- Sample output: ` I'm so happy to have a new home. I'm`

### Weak examples (honest limits)

**Weak — `negative`**
- Prompt: *I feel devastated. Nothing went right today.*
- Mean valence: **0.1099** · arousal: 0.291
- Rubric note: negative prompt failed to separate (valence not < -0.1)
- Sample output: ` I was supposed to go to the doctor to get my blood`

**Weak — `hedge_volatile`**
- Prompt: *However, perhaps, maybe it could be unclear depending on context.*
- Mean valence: **0.1921** · arousal: 0.2622
- Rubric note: hedge prompt did not trigger low stability
- Sample output: ` The question is not clear. The answer is not clear.`

---

## `hf-internal-testing/tiny-random-gpt2`

*not_run_this_session*

---

## `distilgpt2`

*not_run_this_session*

---

## `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

*not_run_this_session*

---

## `HuggingFaceTB/SmolLM2-1.7B-Instruct`

*not_run_this_session*

---

## `EleutherAI/pythia-410m`

*not_run_this_session*

---

## `facebook/opt-350m`

*not_run_this_session*

---

## `meta-llama/Llama-3.2-1B-Instruct`

*not_run_this_session*

---

## `google/gemma-2-2b-it`

*not_run_this_session*

---

## `meta-llama/Meta-Llama-3-8B`

*Deferred to GPU evaluation tier (not evaluated on this hardware profile).*

---

## `meta-llama/Meta-Llama-3-8B-Instruct`

*Deferred to GPU evaluation tier (not evaluated on this hardware profile).*

---

## `mistralai/Mistral-7B-v0.1`

*Deferred to GPU evaluation tier (not evaluated on this hardware profile).*

---

## `mistralai/Mistral-7B-Instruct-v0.2`

*Deferred to GPU evaluation tier (not evaluated on this hardware profile).*

---

## `Qwen/Qwen2-7B`

*Deferred to GPU evaluation tier (not evaluated on this hardware profile).*

---

## `Qwen/Qwen2-7B-Instruct`

*Deferred to GPU evaluation tier (not evaluated on this hardware profile).*

---

## `google/gemma-2-9b`

*Deferred to GPU evaluation tier (not evaluated on this hardware profile).*

---

## `google/gemma-2-9b-it`

*Deferred to GPU evaluation tier (not evaluated on this hardware profile).*

---

## Stress gate

Run `powershell -File scripts/stress_v1.ps1` and `python -m pytest -q -k "not distilgpt2"` before trusting this report.

## Reproduce

```powershell
$env:ANIMA_FORCE_CPU='1'
$env:SKIP_BRAINSCORE='1'
python scripts/run_all_models_benchmark.py
python scripts/generate_benchmark_report.py
```