# Deep waiver GPU status

- Finished: `2026-11-03T18:29:35.445134+00:00`
- Device: `Tesla T4`
- OK: **5** / Failed: **0**
- Rollup: `benchmarks/reports/deep_waiver/deep_waiver_rollup.json`
- Attentions: `0` (0=off under 8-bit default)
- Shutdown: **disabled** for this suite

| Model | Status | Halu composite | Best channel AUROC | Polarity | NaN fused |
|---|---|---:|---:|---|---:|
| `Qwen/Qwen2.5-7B-Instruct` | ok | 0.2061 | 0.794 | anti_aligned | 0.0 |
| `Qwen/Qwen2-7B-Instruct` | ok | 0.2984 | 0.7084 | anti_aligned | 0.0 |
| `mistralai/Mistral-7B-Instruct-v0.2` | ok | 0.4063 | 0.6382 | anti_aligned | 0.0 |
| `Qwen/Qwen2.5-3B-Instruct` | ok | 0.31 | 0.8216 | anti_aligned | 0.0 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | ok | 0.3916 | 0.6354 | anti_aligned | 0.0 |
