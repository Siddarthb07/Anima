"""POC intervention demo — runs emotional prompt suite via REST /generate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "benchmarks" / "fixtures" / "poc_emotional_prompts.json"


def post_generate(
    base: str,
    model: str,
    prompt: str,
    *,
    guard_mode: str = "observe",
    intervention_mode: str = "none",
    max_new_tokens: int = 32,
) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "detect_suppression": True,
            "guard_mode": guard_mode,
            "intervention_mode": intervention_mode,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base.rstrip('/')}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description="Run POC emotional prompts against anima API")
    p.add_argument("--api", default=os.environ.get("ANIMA_API", "http://127.0.0.1:8010"))
    p.add_argument("--model", default=os.environ.get("ANIMA_HERO_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    p.add_argument("--guard-mode", default="gate", choices=["observe", "gate"])
    p.add_argument("--intervention", default="none", choices=["none", "dampen"])
    p.add_argument("--compare-dampen", action="store_true", help="Run intervention_ablation with none vs dampen")
    args = p.parse_args()

    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    prompts = data["prompts"]
    if args.compare_dampen:
        prompts = [x for x in prompts if x["id"] == "intervention_ablation"] or prompts[-1:]

    report = []
    modes = [("none", args.guard_mode)] if not args.compare_dampen else [("none", "observe"), ("dampen", "observe")]

    for intervention_mode, guard_mode in modes:
        for item in prompts:
            try:
                out = post_generate(
                    args.api,
                    args.model,
                    item["text"],
                    guard_mode=guard_mode,
                    intervention_mode=intervention_mode,
                )
            except Exception as exc:
                print(f"FAIL {item['id']} ({intervention_mode}): {exc}", file=sys.stderr)
                return 1
            s = out.get("summary") or {}
            report.append(
                {
                    "id": item["id"],
                    "intervention_mode": intervention_mode,
                    "guard_mode": guard_mode,
                    "mean_valence": s.get("mean_valence"),
                    "stability_score": s.get("stability_score"),
                    "max_valence_swing": s.get("max_valence_swing"),
                    "guard_gated_token_count": s.get("guard_gated_token_count"),
                }
            )
            print(
                f"{item['id']:20} mode={intervention_mode:6} "
                f"V={s.get('mean_valence')} stab={s.get('stability_score')} "
                f"swing={s.get('max_valence_swing')}"
            )

    out_path = ROOT / "benchmarks" / "reports" / "poc_demo_latest.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
