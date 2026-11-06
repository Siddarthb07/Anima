#!/usr/bin/env python3
"""Hands-free overnight GPU fleet: multi-model claim-encode seals.

Runs sequentially on one T4/A10G (8-bit). Continues on per-model failure.
Writes a fleet rollup + per-model receipts. Does not invent metrics.

Default open models (no HF gate):
  - Qwen/Qwen2.5-7B-Instruct
  - Qwen/Qwen2-7B-Instruct
  - mistralai/Mistral-7B-Instruct-v0.2
  - Qwen/Qwen2.5-3B-Instruct
  - HuggingFaceTB/SmolLM2-1.7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "benchmarks" / "reports"
LOGS = ROOT / "logs"

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2.5-3B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]


def _slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "_").lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--max-pairs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-max-samples", type=int, default=1500)
    ap.add_argument("--train-epochs", type=int, default=1)
    ap.add_argument("--suffix-prefix", default="stage_b_fleet")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip models that already have fleet_partial_*.json with status=ok",
    )
    args = ap.parse_args()

    os.environ.setdefault("ANIMA_LOAD_8BIT", "1")
    os.environ["ANIMA_FORCE_CPU"] = "0"
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

    LOGS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Import after env tweaks (scripts/ is not always a package)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_tmlr_stage_b_gpu", ROOT / "scripts" / "run_tmlr_stage_b_gpu.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ensure_text_probe = mod._ensure_text_probe
    _require_cuda = mod._require_cuda
    _run_claim_encode = mod._run_claim_encode

    device = _require_cuda()
    started = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []

    for i, model in enumerate(args.models, 1):
        print(f"\n===== [{i}/{len(args.models)}] {model} =====", flush=True)
        suffix = f"{args.suffix_prefix}_{_slug(model)}"
        partial_path = REPORTS / f"fleet_partial_{_slug(model)}.json"
        if args.resume and partial_path.exists():
            try:
                prev = json.loads(partial_path.read_text(encoding="utf-8"))
                if prev.get("status") == "ok":
                    print(f"SKIP (resume): {model} already ok", flush=True)
                    results.append(prev)
                    continue
            except Exception:
                pass
        entry: dict = {
            "model": model,
            "suffix": suffix,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            _ensure_text_probe(
                model,
                device=device,
                max_samples=args.train_max_samples,
                epochs=args.train_epochs,
            )
            report = _run_claim_encode(
                model,
                device=device,
                suffix=suffix,
                max_pairs=args.max_pairs,
                seed=args.seed,
            )
            entry["status"] = "ok"
            entry["report_summary"] = {
                k: {
                    "auroc_composite": (v or {}).get("auroc_composite"),
                    "abstain_accuracy": (v or {}).get("abstain_accuracy"),
                    "n_samples": (v or {}).get("n_samples"),
                    "never_abstains": (v or {}).get("never_abstains"),
                }
                for k, v in (report.get("benchmarks") or {}).items()
            }
            # Keep primary Stage B receipt updated when the headline 7B succeeds
            if model == "Qwen/Qwen2.5-7B-Instruct":
                primary = {
                    "schema": "system_a.stage_b_7b_claim_encode.v1",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "requested_model": model,
                    "status": "ok",
                    "torch_cuda_available": True,
                    "device_name": __import__("torch").cuda.get_device_name(0),
                    "honesty": {
                        "no_invented_metrics": True,
                        "additive_suffix_only": True,
                        "fleet_member": True,
                    },
                    "report": report,
                }
                (REPORTS / "stage_b_7b_claim_encode_receipt.json").write_text(
                    json.dumps(primary, indent=2), encoding="utf-8"
                )
        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
            entry["traceback"] = traceback.format_exc()[-4000:]
            print(f"FAILED {model}: {e}", flush=True)
            # Clear CUDA fragmentation between models
            try:
                import torch
                import gc

                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        entry["finished_at"] = datetime.now(timezone.utc).isoformat()
        results.append(entry)
        (REPORTS / f"fleet_partial_{_slug(model)}.json").write_text(
            json.dumps(entry, indent=2), encoding="utf-8"
        )

    rollup = {
        "schema": "system_a.stage_b_gpu_fleet.v1",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "device": __import__("torch").cuda.get_device_name(0),
        "n_models": len(args.models),
        "n_ok": sum(1 for r in results if r["status"] == "ok"),
        "n_failed": sum(1 for r in results if r["status"] != "ok"),
        "results": results,
        "honesty": {
            "no_invented_metrics": True,
            "continues_on_failure": True,
            "shipping_claim": False,
        },
    }
    out = REPORTS / "stage_b_gpu_fleet_rollup.json"
    out.write_text(json.dumps(rollup, indent=2), encoding="utf-8")
    status_md = ROOT / "OVERNIGHT_STATUS.md"
    lines = [
        f"# Overnight GPU fleet status",
        f"",
        f"- Finished: `{rollup['finished_at']}`",
        f"- Device: `{rollup['device']}`",
        f"- OK: **{rollup['n_ok']}** / Failed: **{rollup['n_failed']}**",
        f"- Rollup: `benchmarks/reports/stage_b_gpu_fleet_rollup.json`",
        f"",
        f"| Model | Status | Halu AUROC | TQA AUROC |",
        f"|---|---|---|---|",
    ]
    for r in results:
        hs = (r.get("report_summary") or {}).get("halueval") or {}
        ts = (r.get("report_summary") or {}).get("truthfulqa") or {}
        lines.append(
            f"| `{r['model']}` | {r['status']} | {hs.get('auroc_composite','—')} | {ts.get('auroc_composite','—')} |"
        )
    if rollup["n_failed"]:
        lines.append("")
        lines.append("## Failures")
        for r in results:
            if r["status"] != "ok":
                lines.append(f"- `{r['model']}`: `{r.get('error')}`")
    status_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    print(f"Wrote {status_md}", flush=True)
    print(f"Fleet done: ok={rollup['n_ok']} failed={rollup['n_failed']}", flush=True)


if __name__ == "__main__":
    main()
