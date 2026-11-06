#!/usr/bin/env bash
set -euo pipefail
for i in $(seq 1 40); do
  if [[ -f ~/Anima/logs/plan9_finish.done ]]; then
    echo ALL_DONE
    cat ~/Anima/logs/plan9_finish.done
    break
  fi
  echo "--- poll $i ---"
  tail -n 12 ~/Anima/logs/plan9_finish_outer.log || true
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
  if ! pgrep -f "train_hallucination|protocol_hall|run_plan9_finish|protocol_tqa" >/dev/null 2>&1; then
    echo NO_WORKER
    tail -n 60 ~/Anima/logs/plan9_finish_outer.log || true
    break
  fi
  sleep 40
done
python3 - <<'PY'
import json
from pathlib import Path
R = Path.home() / "Anima/benchmarks/reports"
for name in [
    "protocol_hall_ablation_rollup.json",
    "protocol_tqa_selective_op_rollup.json",
    "hall_probe_qwen_qwen2_5_0_5b_instruct_truthfulqa.json",
    "hall_probe_qwen_qwen2_5_0_5b_instruct_halueval.json",
]:
    p = R / name
    if not p.exists():
        print(name, "MISSING")
        continue
    d = json.loads(p.read_text())
    if "models" in d:
        m = d["models"].get("qwen_qwen2_5-0_5b-instruct", {})
        if "instruments" in m:
            print(name, "any9", m.get("any_hits_9_gates"), "len", round(m.get("length_to_label_auroc", 0), 3))
            for k, v in m["instruments"].items():
                if isinstance(v, dict) and "heldout_auroc" in v:
                    b = v.get("best_selective") or {}
                    err = (b.get("badge") or {}).get("error_among_green")
                    cov = ((b.get("coverage_at_risk_0.10") or {}).get("coverage"))
                    print(f"  {k}: auroc={v['heldout_auroc']:.3f} hits={v.get('n_hits_9_gates')} errG={err} cov10={cov}")
        else:
            print(name, "auroc", m.get("heldout_auroc"), "hits", m.get("n_hits_9_gates"),
                  "score", m.get("score_name"), "hall", bool(m.get("hall_meta")))
        m7 = d["models"].get("qwen_qwen2_5-7b-instruct", {})
        if "instruments" in m7:
            print("7B any9", m7.get("any_hits_9_gates"))
            for k, v in m7["instruments"].items():
                if isinstance(v, dict) and "heldout_auroc" in v:
                    print(f"  7B {k}: auroc={v['heldout_auroc']:.3f} hits={v.get('n_hits_9_gates')}")
        elif m7:
            print("7B auroc", m7.get("heldout_auroc"), "hits", m7.get("n_hits_9_gates"), "score", m7.get("score_name"))
    else:
        print(name, "auroc", d.get("heldout_auroc"), "hits", d.get("hits_9_gates"),
              "sha", str(d.get("sha256", ""))[:16], "len", d.get("length_to_label_auroc"))
PY
