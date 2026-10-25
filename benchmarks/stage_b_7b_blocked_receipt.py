#!/usr/bin/env python3
"""Seal an honest blocked receipt when Stage B 7B claim-encode cannot run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

REPORTS = Path(__file__).resolve().parent / "reports"


def main() -> None:
    cuda = bool(torch.cuda.is_available())
    out = {
        "schema": "anima.stage_b_7b_claim_encode.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_model": "Qwen/Qwen2.5-7B-Instruct",
        "status": "blocked_no_cuda" if not cuda else "ready_to_run",
        "torch_cuda_available": cuda,
        "torch_version": getattr(torch, "__version__", None),
        "honesty": {
            "no_invented_metrics": True,
            "run_when_gpu": "python scripts/run_tmlr_stage_b_gpu.py --mode claim-encode --model Qwen/Qwen2.5-7B-Instruct --suffix stage_b_7b",
        },
        "note": (
            "This machine has no usable CUDA device for a sealed 7B claim-encode run. "
            "Do not invent placeholder AUROCs. Re-run on a GPU host and replace this receipt."
        ),
    }
    path = REPORTS / "stage_b_7b_claim_encode_receipt.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
