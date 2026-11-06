#!/usr/bin/env python3
"""Local helper: normalize + scp plan9 finish scripts, start remote job."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(r"C:\Users\siddu\Projects\Anima")
FILES = [
    ("benchmarks/train_hallucination_probe.py", ROOT / "benchmarks/train_hallucination_probe.py"),
    ("benchmarks/protocol_hall_ablation.py", ROOT / "benchmarks/protocol_hall_ablation.py"),
    ("benchmarks/protocol_tqa_selective_op.py", ROOT / "benchmarks/protocol_tqa_selective_op.py"),
    ("benchmarks/protocol_second_channel_probe.py", ROOT / "benchmarks/protocol_second_channel_probe.py"),
    ("benchmarks/protocol_phase2_9_10_battery.py", ROOT / "benchmarks/protocol_phase2_9_10_battery.py"),
    ("scripts/run_plan9_finish.sh", ROOT / "scripts/run_plan9_finish.sh"),
]

POLL = r'''#!/usr/bin/env bash
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
'''


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    for dst, src in FILES:
        text = src.read_text(encoding="utf-8")
        if text.startswith("\ufeff"):
            text = text[1:]
        text = text.replace("\r\n", "\n")
        local = tmp / Path(dst).name
        local.write_bytes(text.encode("utf-8"))
        subprocess.check_call(["scp", str(local), f"anima-gpu:~/Anima/{dst}"])
        print("uploaded", dst)

    poll = tmp / "poll_finish.sh"
    poll.write_bytes(POLL.replace("\r\n", "\n").encode("utf-8"))
    subprocess.check_call(["scp", str(poll), "anima-gpu:~/poll_finish.sh"])

    subprocess.check_call([
        "ssh", "anima-gpu",
        "chmod +x ~/Anima/scripts/run_plan9_finish.sh; "
        "rm -f ~/Anima/logs/plan9_finish.done; "
        "nohup bash ~/Anima/scripts/run_plan9_finish.sh > ~/Anima/logs/plan9_finish_outer.log 2>&1 & "
        "echo STARTED $!",
    ])
    subprocess.check_call(["ssh", "anima-gpu", "bash ~/poll_finish.sh"])


if __name__ == "__main__":
    main()
