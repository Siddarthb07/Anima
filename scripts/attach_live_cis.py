"""Offline: attach bootstrap AUROC CIs to existing *_live_*.json and live_guard_*.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from benchmarks.guard_metrics import bootstrap_auroc_ci, score_abstain_accuracy

REPORTS = Path(__file__).resolve().parent.parent / "benchmarks" / "reports"


def refresh_row_file(path: Path, *, label_key: str, seed: int = 42) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    metrics = score_abstain_accuracy(rows, label_key=label_key, with_ci=True, ci_seed=seed)
    data["metrics"] = metrics
    data["protocol"] = data.get("protocol") or "L"
    data["seed"] = seed
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return metrics


def refresh_live_guard(slug: str, reports: Path = REPORTS, seed: int = 42) -> None:
    lg_path = reports / f"live_guard_{slug}.json"
    report = json.loads(lg_path.read_text(encoding="utf-8")) if lg_path.is_file() else {
        "status": "ok",
        "mode": "live_lm_claim_encoding",
        "benchmarks": {},
    }
    for suite, label_key in (("halueval", "is_hallucination"), ("truthfulqa", "should_abstain")):
        row_path = reports / f"{suite}_live_{slug}.json"
        if not row_path.is_file():
            continue
        metrics = refresh_row_file(row_path, label_key=label_key, seed=seed)
        report.setdefault("benchmarks", {})[suite] = {
            "status": "ok",
            "source": "live",
            "protocol": "L",
            **metrics,
        }
        ci = metrics.get("auroc_ci95") or {}
        print(f"  {suite}: AUROC={metrics['auroc_composite']} CI=[{ci.get('lo')}, {ci.get('hi')}]")
    lg_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {lg_path}")


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--slug", required=True)
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    refresh_live_guard(args.slug, args.reports_dir, seed=args.seed)


if __name__ == "__main__":
    main()
