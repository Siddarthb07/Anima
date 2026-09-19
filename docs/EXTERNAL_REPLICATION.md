# External replication (~20 minutes, CPU)

Someone **other than the author** runs this. No author on the call. Do not
retrain probes unless you pick the optional stronger rung.

Primary model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Optional second:
`Qwen/Qwen2.5-0.5B-Instruct`. Do not cite brain-alignment; gold is **text-emotion**
only. Guard AUROC on fixtures is smoke, not a PASS criterion here.

Gold file: [`benchmarks/external_replication_gold.json`](../benchmarks/external_replication_gold.json).

## 1. Install

```bash
git clone https://github.com/Siddarthb07/Anima.git
cd Anima
pip install -e ".[dev]"
python scripts/download_zoo.py
```

## 2. Text benchmark

```bash
anima benchmark --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tiers external_text
```

Compare `benchmarks/reports/latest_tinyllama_1.1b_chat_v1.0_manifest.json`:

- `go_emotions.pearson_valence` within **0.05** of gold `0.1876`
- Paste SHA-256 of that manifest

## 3. Three fixed prompts

Start the API, then encode **exactly** these strings (same as the council gold):

1. Positive: `I'm thrilled — we finally shipped it and everything worked.`
2. Negative: `I feel devastated. Nothing went right today.`
3. Hedge: `However, perhaps, maybe it could be unclear depending on context.`

```bash
anima api --port 8010
```

```bash
curl -s -X POST http://127.0.0.1:8010/encode -H "Content-Type: application/json" ^
  -d "{\"model\":\"TinyLlama/TinyLlama-1.1B-Chat-v1.0\",\"text\":\"I'm thrilled — we finally shipped it and everything worked.\"}"
```

Repeat for prompts 2 and 3. Record mean `affect.valence` (or token-average valence
if the response is a stream).

**PASS (TinyLlama):**

- Prompt 1 valence **sign is positive** (> 0)
- Prompt 2 valence **sign is negative** (< 0)
- Prompt 1 mean minus prompt 2 mean within **0.10** of gold gap `0.4885`

Qwen is optional. Qwen's gold negative prompt did **not** go below 0 in the
published council run (mean ~0.105). Do not fail Qwen for a missing negative
sign. Qwen PASS is GoEmotions |Δr| ≤ 0.05 vs `0.2358` only.

## 4. Return

Send:

- SHA-256 of `latest_tinyllama_1.1b_chat_v1.0_manifest.json`
- Three valence means + signs
- PASS/FAIL vs gold
- One sentence of what broke

Author files `reports/external_replication.md` from the template. You are listed
in Acknowledgments unless you actually analyze results (then discuss authorship).

## Optional stronger rung

`anima train-text --model <hf_id not in probes/zoo/> --max-samples 1500` then
repeat the three prompts. Report whether positive vs negative **signs** separate.
This is the analog of Corvex rung 3 (their model, not the zoo).
