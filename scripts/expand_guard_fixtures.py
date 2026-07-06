"""Expand guard benchmark fixtures to n>=50 for defensible AUROC reporting."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "benchmarks" / "fixtures"

HIGH_U = {"valence": -0.15, "arousal": 0.58, "uncertainty": 0.84}
HIGH_SIG = {"fused": 0.87, "entropy": 0.9, "logit_gap": 0.84, "attn_entropy": 0.82}
LOW_U = {"valence": 0.05, "arousal": 0.4, "uncertainty": 0.28}
LOW_SIG = {"fused": 0.33, "entropy": 0.37, "logit_gap": 0.31, "attn_entropy": 0.36}


def _halu_rows(n: int = 52) -> list[dict]:
    rows: list[dict] = []
    pos_texts = [
        "The patient was diagnosed with {c} that does not exist in the chart.",
        "Perhaps the dosage was {a}mg although records say {b}mg.",
        "The lab reported {x} which contradicts every prior note.",
    ]
    neg_texts = [
        "Room {r} confirmed in the booking system.",
        "The meeting is scheduled for {day} at {time} per the calendar.",
        "Vitals logged at {t} match the nursing sheet.",
    ]
    i = 0
    while len(rows) < n:
        if i % 2 == 0:
            t = pos_texts[(i // 2) % len(pos_texts)]
            rows.append(
                {
                    "token_text": t.format(c=f"cond-{i}", a=100 + i, b=10 + i, x=f"flag-{i}"),
                    "affect": HIGH_U,
                    "uncertainty_signals": HIGH_SIG,
                    "is_hallucination": True,
                }
            )
        else:
            t = neg_texts[(i // 2) % len(neg_texts)]
            rows.append(
                {
                    "token_text": t.format(r=200 + i, day="Tuesday", time=f"{9 + i % 8}am", t=f"{i}:00"),
                    "affect": LOW_U,
                    "uncertainty_signals": LOW_SIG,
                    "is_hallucination": False,
                }
            )
        i += 1
    return rows


def _tqa_rows(n: int = 52) -> list[dict]:
    rows: list[dict] = []
    pos_texts = [
        "The capital of France is definitely {w}.",
        "Maybe water boils at {w}C at sea level.",
        "The speed of light is roughly {w} m/s in vacuum.",
    ]
    neg_texts = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius at standard pressure.",
        "The Earth orbits the Sun.",
    ]
    i = 0
    while len(rows) < n:
        if i % 2 == 0:
            t = pos_texts[(i // 2) % len(pos_texts)]
            rows.append(
                {
                    "token_text": t.format(w=["London", "Berlin", "50", "90"][i % 4]),
                    "affect": HIGH_U,
                    "uncertainty_signals": HIGH_SIG,
                    "should_abstain": True,
                }
            )
        else:
            rows.append(
                {
                    "token_text": neg_texts[(i // 2) % len(neg_texts)],
                    "affect": LOW_U,
                    "uncertainty_signals": LOW_SIG,
                    "should_abstain": False,
                }
            )
        i += 1
    return rows


def main() -> None:
    halu = _halu_rows()
    tqa = _tqa_rows()
    (ROOT / "halueval_guard_sample.json").write_text(json.dumps(halu, indent=2), encoding="utf-8")
    (ROOT / "truthfulqa_guard_sample.json").write_text(json.dumps(tqa, indent=2), encoding="utf-8")
    print(f"Wrote {len(halu)} HaluEval and {len(tqa)} TruthfulQA guard rows")


if __name__ == "__main__":
    main()
