"""
Write BENCHMARK_LOCK.md with SHA-256 checksums of locked eval artifacts.
Offline checks + optional pointer to pytest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPORTS = Path(__file__).resolve().parent.parent / "benchmarks" / "reports"

LOCK_PATTERNS = (
    "halueval_live_*.json",
    "truthfulqa_live_*.json",
    "live_guard_*.json",
    "polarity_diagnostic_*.json",
    "protocol_r_*.json",
    "protocol_r_prime_*.json",
    "semantic_entropy_matched_*.json",
    "guard_ablation_*.json",
    "trust_integrity_*.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(reports: Path) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    seen: set[Path] = set()
    for pat in LOCK_PATTERNS:
        for p in sorted(reports.glob(pat)):
            if p in seen:
                continue
            seen.add(p)
            rows.append((p.name, sha256_file(p), p.stat().st_size))
    return rows


def write_lock(reports: Path = REPORTS) -> Path:
    rows = collect(reports)
    lines = [
        "# Anima eval freeze (TMLR / AICS)",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "IEEE honesty: cite Protocol L live AUROC + CIs; fixture Protocol F is smoke only;",
        "Protocol R is optimistic secondary; Protocol R' is held-out primary for thresholds.",
        "",
        "| File | Bytes | SHA-256 |",
        "|------|------:|---------|",
    ]
    for name, digest, size in rows:
        lines.append(f"| `{name}` | {size} | `{digest}` |")
    lines.extend(
        [
            "",
            "## Checklist",
            "",
            "- [ ] A0 polarity diagnostic logged",
            "- [ ] Qwen regression (AUROC/CI/R) matches paper_numbers",
            "- [ ] TinyLlama F/L/R + CIs frozen",
            "- [ ] Matched SE table (both suites, both models)",
            "- [ ] R' held-out for both models",
            "- [ ] Ablations both models",
            "- [ ] paper_numbers.json synced",
            "- [ ] `pytest tests/test_eval_lift_a0_a5.py` green",
            "",
        ]
    )
    dest = reports / "BENCHMARK_LOCK.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": [{"name": n, "sha256": d, "bytes": s} for n, d, s in rows],
    }
    (reports / "BENCHMARK_LOCK.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dest


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Freeze Anima eval artifacts")
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    args = p.parse_args(argv)
    dest = write_lock(args.reports_dir)
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
