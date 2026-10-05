# A0 Polarity diagnostic

Generated: `2026-10-23T11:18:55.975858+00:00`
Model slug: `qwen_qwen2_5-0_5b-instruct`

Production HaluEval label wiring is **unchanged** (correct→False / hallucinated→True). This offline invert tests whether AUROC < 0.5 is anti-aligned ranking vs a label bug.

## halueval

- n=80 (pos=40, neg=40)
- AUROC original: **0.2472** CI [0.1445, 0.363]
- AUROC flipped: **0.7528** (expected if anti-aligned: 0.7528)
- mean score label=1: 0.4399; label=0: 0.4584
- **verdict:** `anti_aligned_ranking` — AUROC=0.2472 < 0.45 and flipped≈1−original (0.7528): score ranks opposite the intended positive class; not a label-wire bug.

## truthfulqa

- n=80 (pos=40, neg=40)
- AUROC original: **0.4587** CI [0.3277, 0.5917]
- AUROC flipped: **0.5413** (expected if anti-aligned: 0.5413)
- mean score label=1: 0.4405; label=0: 0.4438
- **verdict:** `near_chance_coherent` — AUROC=0.4587 near chance; flip≈1−original confirms coherent ranking (not a label bug), but not usable for trust UI.

## IEEE / TMLR integrity

- Do not silently flip labels in `live_guard_eval` to manufacture AUROC > 0.5.
- Cite live AUROC with CI; lead with collapse, not fixture 1.00.
