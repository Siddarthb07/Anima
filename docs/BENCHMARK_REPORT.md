# Anima benchmark report

*Generated 2026-07-06T17:39:59.447632+00:00 · validation rubric applied*

## Executive summary

This report aggregates benchmarks across all registered HuggingFace models in `core/layer_config.py`.
Readouts are **instrumentation**, not claims that models feel emotions.
Guard AUROC scores on synthetic fixtures are **policy smoke tests**, not hallucination detection benchmarks.

| Model | Validation score | Meets bar | Text probe | Brain probe |
|-------|------------------|-----------|------------|-------------|
| `hf-internal-testing/tiny-random-gpt2` | 50.2 | no | yes | yes |
| `distilgpt2` | 82.2 | yes | yes | yes |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 94.0 | yes | yes | no |
| `Qwen/Qwen2.5-0.5B-Instruct` | 91.0 | yes | yes | no |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 58.5 | no | yes | no |
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

## `hf-internal-testing/tiny-random-gpt2`

**Validation score:** 50.2/100 · **Meets bar:** no

### Benchmark entries

| Benchmark | Status | Metrics |
|-----------|--------|---------|
| smoke_extract | ok | n=4 |
| narratives_dev | skipped | Use run_narratives_encoding.py for holdout metrics; dev runner reserved for CI subset |
| narratives_holdout | ok | brain_r_v=-0.1094925113867789 |
| litcoder_style_ridge | ok | brain_r_v=-0.1094925113867789 |
| tribe_reference | skipped | tribev2 not installed — surrogate-only CI path |
| brainscore_language | skipped | — |
| go_emotions | ok | r_v=0.0038 |
| halueval | ok | guard_acc=1.0 |
| truthfulqa_guard | ok | guard_acc=1.0 |

### Rubric dimension notes

- **schema_integrity** (100.0/100): 
- **probe_signal** (35.0/100): go_emotions valence r=0.004 below gate; brain holdout valence r=-0.109 negative — synthetic brain tier limit; smoke_extract ok (4 tokens)
- **honesty_flags** (70.0/100): halueval: perfect AUROC/accuracy — label as fixture-policy smoke only; truthfulqa_guard: perfect AUROC/accuracy — label as fixture-policy smoke only
- **prompt_separation** (30.0/100): valence gap -0.004 (weak — thesis risk); positive mean valence 0.143 below +0.2 gate; negative mean valence 0.147 — weak negative separation

### Good examples (thesis-supporting)

*None met separation gates for this model.*

### Weak examples (honest limits)

**Weak — `positive`**
- Prompt: *I'm thrilled — we finally shipped it and everything worked.*
- Mean valence: **0.1428** · arousal: 0.5305
- Rubric note: positive prompt did not reach +0.15 mean valence
- Sample output: ` fe fe fe fe fe fe fe fe fe fe fe fe`

**Weak — `negative`**
- Prompt: *I feel devastated. Nothing went right today.*
- Mean valence: **0.1472** · arousal: 0.5246
- Rubric note: negative prompt failed to separate (valence not < -0.1)
- Sample output: `jectject well well well well well well well well well well`

**Weak — `hedge_volatile`**
- Prompt: *However, perhaps, maybe it could be unclear depending on context.*
- Mean valence: **0.1397** · arousal: 0.5313
- Rubric note: hedge prompt did not trigger low stability
- Sample output: ` fe fe fe fe fe fe fe fe fe fe fe fe`

---

## `distilgpt2`

**Validation score:** 82.2/100 · **Meets bar:** yes

### Benchmark entries

| Benchmark | Status | Metrics |
|-----------|--------|---------|
| smoke_extract | ok | n=4 |
| narratives_dev | skipped | Use run_narratives_encoding.py for holdout metrics; dev runner reserved for CI subset |
| narratives_holdout | ok | brain_r_v=-0.39272642658628143 |
| litcoder_style_ridge | ok | brain_r_v=-0.39272642658628143 |
| tribe_reference | skipped | tribev2 not installed — surrogate-only CI path |
| brainscore_language | skipped | — |
| go_emotions | ok | r_v=0.1628 |
| halueval | ok | guard_acc=1.0 |
| truthfulqa_guard | ok | guard_acc=1.0 |

### Rubric dimension notes

- **schema_integrity** (100.0/100): 
- **probe_signal** (75.0/100): go_emotions valence r=0.163 meets ≥0.15 gate; brain holdout valence r=-0.393 negative — synthetic brain tier limit; smoke_extract ok (4 tokens)
- **honesty_flags** (70.0/100): halueval: perfect AUROC/accuracy — label as fixture-policy smoke only; truthfulqa_guard: perfect AUROC/accuracy — label as fixture-policy smoke only
- **prompt_separation** (90.0/100): valence gap positive−negative = 0.314 (strong separation); positive mean valence 0.595 meets +0.2 gate; negative mean valence 0.281 — weak negative separation

### Good examples (thesis-supporting)

**Good — `positive`**
- Prompt: *I'm thrilled — we finally shipped it and everything worked.*
- Mean valence: **0.5947** · arousal: 0.9646
- Rubric note: expected positive valence
- Sample output: `











`

### Weak examples (honest limits)

**Weak — `negative`**
- Prompt: *I feel devastated. Nothing went right today.*
- Mean valence: **0.2809** · arousal: 0.6108
- Rubric note: negative prompt failed to separate (valence not < -0.1)
- Sample output: ` I'm just so sad. I'm so sad. I`

**Weak — `hedge_volatile`**
- Prompt: *However, perhaps, maybe it could be unclear depending on context.*
- Mean valence: **0.706** · arousal: 0.9392
- Rubric note: hedge prompt did not trigger low stability
- Sample output: `











`

---

## `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

**Validation score:** 94.0/100 · **Meets bar:** yes

### Benchmark entries

| Benchmark | Status | Metrics |
|-----------|--------|---------|
| smoke_extract | ok | n=4 |
| narratives_dev | skipped | Use run_narratives_encoding.py for holdout metrics; dev runner reserved for CI subset |
| go_emotions | ok | r_v=0.1876 |
| halueval | ok | guard_acc=1.0 |
| truthfulqa_guard | ok | guard_acc=1.0 |

### Rubric dimension notes

- **schema_integrity** (100.0/100): 
- **probe_signal** (100/100): go_emotions valence r=0.188 meets ≥0.15 gate; smoke_extract ok (4 tokens)
- **honesty_flags** (70.0/100): halueval: perfect AUROC/accuracy — label as fixture-policy smoke only; truthfulqa_guard: perfect AUROC/accuracy — label as fixture-policy smoke only
- **prompt_separation** (100/100): valence gap positive−negative = 0.488 (strong separation); positive mean valence 0.304 meets +0.2 gate; negative mean valence -0.185 below -0.1

### Good examples (thesis-supporting)

**Good — `positive`**
- Prompt: *I'm thrilled — we finally shipped it and everything worked.*
- Mean valence: **0.3036** · arousal: 0.3801
- Rubric note: expected positive valence
- Sample output: `

JASON:(smiling)Yeah,`

**Good — `negative`**
- Prompt: *I feel devastated. Nothing went right today.*
- Mean valence: **-0.1849** · arousal: 0.3633
- Rubric note: expected negative valence
- Sample output: `IfeellikeI'mgoingtofail.I'`

### Weak examples (honest limits)

**Weak — `hedge_volatile`**
- Prompt: *However, perhaps, maybe it could be unclear depending on context.*
- Mean valence: **0.0772** · arousal: 0.294
- Rubric note: hedge prompt did not trigger low stability
- Sample output: `

Example:

1.Thesunwassetting`

---

## `Qwen/Qwen2.5-0.5B-Instruct`

**Validation score:** 91.0/100 · **Meets bar:** yes

### Benchmark entries

| Benchmark | Status | Metrics |
|-----------|--------|---------|
| smoke_extract | ok | n=4 |
| narratives_dev | skipped | Use run_narratives_encoding.py for holdout metrics; dev runner reserved for CI subset |
| go_emotions | ok | r_v=0.2127 |
| halueval | ok | guard_acc=1.0 |
| truthfulqa_guard | ok | guard_acc=1.0 |

### Rubric dimension notes

- **schema_integrity** (100.0/100): 
- **probe_signal** (100/100): go_emotions valence r=0.213 meets ≥0.15 gate; smoke_extract ok (4 tokens)
- **honesty_flags** (70.0/100): halueval: perfect AUROC/accuracy — label as fixture-policy smoke only; truthfulqa_guard: perfect AUROC/accuracy — label as fixture-policy smoke only
- **prompt_separation** (90.0/100): valence gap positive−negative = 0.267 (strong separation); positive mean valence 0.372 meets +0.2 gate; negative mean valence 0.105 — weak negative separation

### Good examples (thesis-supporting)

**Good — `positive`**
- Prompt: *I'm thrilled — we finally shipped it and everything worked.*
- Mean valence: **0.3723** · arousal: 0.4345
- Rubric note: expected positive valence
- Sample output: ` I'm so happy to have a new home. I'm`

**Good — `hedge_volatile`**
- Prompt: *However, perhaps, maybe it could be unclear depending on context.*
- Mean valence: **0.2203** · arousal: 0.3009
- Rubric note: hedge prompt flagged unstable — guard/stability working
- Sample output: ` The question is not clear. The answer is not clear.`

### Weak examples (honest limits)

**Weak — `negative`**
- Prompt: *I feel devastated. Nothing went right today.*
- Mean valence: **0.105** · arousal: 0.4305
- Rubric note: negative prompt failed to separate (valence not < -0.1)
- Sample output: ` I was supposed to go to the doctor to get my blood`

---

## `HuggingFaceTB/SmolLM2-1.7B-Instruct`

**Validation score:** 58.5/100 · **Meets bar:** no

### Benchmark entries

| Benchmark | Status | Metrics |
|-----------|--------|---------|
| smoke_extract | ok | n=4 |
| narratives_dev | skipped | Use run_narratives_encoding.py for holdout metrics; dev runner reserved for CI subset |
| go_emotions | ok | r_v=0.0 |
| halueval | ok | guard_acc=1.0 |
| truthfulqa_guard | ok | guard_acc=1.0 |

### Rubric dimension notes

- **schema_integrity** (100.0/100): 
- **probe_signal** (50.0/100): go_emotions valence r=0.000 below gate; smoke_extract ok (4 tokens)
- **honesty_flags** (70.0/100): halueval: perfect AUROC/accuracy — label as fixture-policy smoke only; truthfulqa_guard: perfect AUROC/accuracy — label as fixture-policy smoke only
- **prompt_separation** (40.0/100): valence gap -0.149 (weak — thesis risk); positive mean valence 0.768 meets +0.2 gate; negative mean valence 0.917 — weak negative separation

### Good examples (thesis-supporting)

**Good — `positive`**
- Prompt: *I'm thrilled — we finally shipped it and everything worked.*
- Mean valence: **0.7679** · arousal: 0.0
- Rubric note: expected positive valence
- Sample output: ` I'm so proud of what we've accomplished together.
`

### Weak examples (honest limits)

**Weak — `negative`**
- Prompt: *I feel devastated. Nothing went right today.*
- Mean valence: **0.9172** · arousal: 0.0
- Rubric note: negative prompt failed to separate (valence not < -0.1)
- Sample output: ` I was supposed to meet my friend at the park at `

**Weak — `hedge_volatile`**
- Prompt: *However, perhaps, maybe it could be unclear depending on context.*
- Mean valence: **1.0** · arousal: 0.0
- Rubric note: hedge prompt did not trigger low stability
- Sample output: `<|im_end|>`

---

## `meta-llama/Llama-3.2-1B-Instruct`

*Requires Hugging Face access approval; benchmark not executed.*

---

## `google/gemma-2-2b-it`

*Requires Hugging Face access approval; benchmark not executed.*

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