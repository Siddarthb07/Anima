"""Verify A0–A5 IEEE eval artifacts exist and key numbers are consistent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPORTS = Path(__file__).resolve().parent.parent / "benchmarks" / "reports"

REQUIRED = [
    "polarity_diagnostic_qwen_qwen2_5-0_5b-instruct.json",
    "live_guard_qwen_qwen2_5-0_5b-instruct.json",
    "live_guard_tinyllama_tinyllama-1_1b-chat-v1_0.json",
    "protocol_r_qwen_qwen2_5-0_5b-instruct.json",
    "protocol_r_tinyllama_tinyllama-1_1b-chat-v1_0.json",
    "protocol_r_prime_qwen_qwen2_5-0_5b-instruct.json",
    "protocol_r_prime_tinyllama_tinyllama-1_1b-chat-v1_0.json",
    "guard_ablation_qwen_qwen2_5-0_5b-instruct.json",
    "guard_ablation_tinyllama_tinyllama-1_1b-chat-v1_0.json",
    "fixture_guard_tinyllama_tinyllama-1_1b-chat-v1_0.json",
    "semantic_entropy_matched_qwen_qwen2_5-0_5b-instruct.json",
    "semantic_entropy_matched_tinyllama_tinyllama-1_1b-chat-v1_0.json",
    "BENCHMARK_LOCK.md",
]


def _load(name: str) -> Optional[dict[str, Any]]:
    p = REPORTS / name
    if not p.is_file():
        return None
    if p.suffix == ".md":
        return {"_exists": True, "bytes": p.stat().st_size}
    return json.loads(p.read_text(encoding="utf-8"))


def verify() -> dict[str, Any]:
    missing: list[str] = []
    checks: list[dict[str, Any]] = []
    for name in REQUIRED:
        data = _load(name)
        ok = data is not None
        if not ok:
            missing.append(name)
        checks.append({"file": name, "present": ok})

    # Locked Qwen live AUROC
    lg = _load("live_guard_qwen_qwen2_5-0_5b-instruct.json") or {}
    h = (lg.get("benchmarks") or {}).get("halueval") or {}
    t = (lg.get("benchmarks") or {}).get("truthfulqa") or {}
    if h:
        checks.append(
            {
                "check": "qwen_halueval_auroc",
                "ok": abs(float(h.get("auroc_composite", -1)) - 0.2469) < 1e-3,
                "value": h.get("auroc_composite"),
                "expected": 0.2469,
            }
        )
        checks.append(
            {
                "check": "qwen_halueval_never_abstains",
                "ok": bool(h.get("never_abstains") or (h.get("tp", 1) == 0 and h.get("fp", 1) == 0)),
                "tp": h.get("tp"),
                "fp": h.get("fp"),
            }
        )
    if t:
        checks.append(
            {
                "check": "qwen_truthfulqa_auroc",
                "ok": abs(float(t.get("auroc_composite", -1)) - 0.4587) < 1e-3,
                "value": t.get("auroc_composite"),
                "expected": 0.4587,
            }
        )

    pol = _load("polarity_diagnostic_qwen_qwen2_5-0_5b-instruct.json") or {}
    hal_pol = (pol.get("suites") or {}).get("halueval") or {}
    if hal_pol:
        checks.append(
            {
                "check": "a0_anti_aligned",
                "ok": hal_pol.get("verdict") == "anti_aligned_ranking",
                "verdict": hal_pol.get("verdict"),
                "auroc": hal_pol.get("auroc_original"),
                "flip": hal_pol.get("auroc_flipped"),
            }
        )

    ab = _load("guard_ablation_qwen_qwen2_5-0_5b-instruct.json") or {}
    abl = ((ab.get("halueval") or {}).get("ablations") or {})
    if abl:
        full = (abl.get("full") or {}).get("auroc_composite")
        probe = (abl.get("probe_only") or {}).get("auroc_composite")
        checks.append(
            {
                "check": "fused_hurts_halueval",
                "ok": probe is not None and full is not None and float(probe) > float(full),
                "full": full,
                "probe_only": probe,
            }
        )

    se = _load("semantic_entropy_matched_qwen_qwen2_5-0_5b-instruct.json") or {}
    se_h = ((se.get("suites") or {}).get("halueval") or {}).get("baselines") or {}
    if se_h:
        checks.append(
            {
                "check": "se_table_has_baselines",
                "ok": all(k in se_h for k in ("anima_composite", "probe_only", "semantic_entropy", "claim_disagreement")),
                "keys": sorted(se_h.keys()),
            }
        )

    failed = [c for c in checks if c.get("present") is False or c.get("ok") is False]
    return {
        "status": "ok" if not missing and not failed else "incomplete",
        "missing_files": missing,
        "failed_checks": failed,
        "checks": checks,
        "n_required": len(REQUIRED),
        "n_present": len(REQUIRED) - len(missing),
    }


def main(argv: Optional[list[str]] = None) -> None:
    argparse.ArgumentParser(description="Verify A0–A5 eval completeness").parse_args(argv)
    report = verify()
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
