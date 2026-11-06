#!/usr/bin/env bash
# Re-run C1 after text probe is present (B1 already sealed).
set -euo pipefail
cd "$HOME/Anima"
source .venv/bin/activate
export ANIMA_FORCE_CPU=0 ANIMA_OUTPUT_ATTENTIONS=0
export HF_HOME="${HF_HOME:-$HOME/Anima/.hf_cache}"
mkdir -p logs benchmarks/reports

echo "[C1 redo] Length-balanced HaluEval encode n=300 with text probe"
python - <<'PY'
from benchmarks.length_balanced_pairs import build_halueval_balanced
from benchmarks.live_guard_eval import evaluate_pairs
from benchmarks.guard_metrics import score_abstain_accuracy
import json
from pathlib import Path
pairs = build_halueval_balanced(max_pairs=300, max_ratio=1.25, seed=42)
print("built", len(pairs))
rows = evaluate_pairs("Qwen/Qwen2.5-0.5B-Instruct", pairs, label_key="is_hallucination", device="cuda")
# sanity
n_nan = sum(1 for r in rows if r.get("composite_score") != r.get("composite_score"))
origins = {((r.get("_run_meta") or {}).get("probe_meta") or {}).get("probe_origin") for r in rows}
print("n_nan_composite", n_nan, "probe_origins", origins)
metrics = score_abstain_accuracy(rows, label_key="is_hallucination", with_ci=True, ci_B=2000, ci_seed=42)
out = Path("benchmarks/reports/halueval_live_qwen_qwen2_5-0_5b-instruct_lenbal_n300.json")
out.write_text(json.dumps({"rows": rows, "metrics": metrics, "protocol": "L_length_balanced_v1", "seed": 42}, indent=2))
print("wrote", out, "n", metrics.get("n_samples"), "auroc", metrics.get("auroc_composite"))
print(json.dumps(metrics, indent=2, default=str)[:2000])
PY

echo "[B2] Transfer gap"
python -m benchmarks.protocol_transfer_gap 2>&1 | tee logs/phase_a_transfer_gap.log || true
date -u > logs/phase_c1_redo.done
echo DONE
