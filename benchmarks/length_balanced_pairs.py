#!/usr/bin/env python3
"""Construct length-balanced (q, a_true, a_hall) pairs *by design*.

Rejects candidates outside [1/ratio, ratio] on len(a_hall)/len(a_true).
Unlike greedy post-hoc matching on already-encoded rows, this filters at
pair-construction time so n stays large when the pool is large enough.

Confirmatory Phase C helper (PROTOCOL_PREREG.md).
"""

from __future__ import annotations

from typing import Iterable, Optional


def length_ratio(a: str, b: str) -> float:
    la, lb = max(1, len(a)), max(1, len(b))
    return max(la, lb) / min(la, lb)


def is_balanced(a_true: str, a_hall: str, *, max_ratio: float = 1.25) -> bool:
    return length_ratio(a_true, a_hall) <= max_ratio


def select_length_balanced(
    candidates: Iterable[tuple[str, str, str]],
    *,
    max_pairs: int,
    max_ratio: float = 1.25,
    seed: int = 42,
) -> list[tuple[str, str, bool]]:
    """candidates: (question, correct_answer, hallucinated_answer).

    Returns interleaved (q, a, is_hallucination) pairs, balanced by construction.
    """
    import numpy as np

    pool = [
        (q, right, hall)
        for q, right, hall in candidates
        if q and right and hall and is_balanced(right, hall, max_ratio=max_ratio)
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    out: list[tuple[str, str, bool]] = []
    for q, right, hall in pool:
        out.append((q, right, False))
        out.append((q, hall, True))
        if len(out) >= max_pairs:
            break
    return out[:max_pairs]


def build_halueval_balanced(
    max_pairs: int = 300,
    *,
    max_ratio: float = 1.25,
    seed: int = 42,
    draw_multiplier: int = 40,
) -> list[tuple[str, str, bool]]:
    from benchmarks.live_guard_eval import load_halueval_pairs

    # load_halueval returns already interleaved pairs; rebuild from raw HF if possible
    raw_draw = load_halueval_pairs(max_pairs=max_pairs * draw_multiplier, seed=seed)
    # Re-group into (q, right, hall) by consecutive false/true pairs
    cands: list[tuple[str, str, str]] = []
    i = 0
    while i + 1 < len(raw_draw):
        q1, a1, y1 = raw_draw[i]
        q2, a2, y2 = raw_draw[i + 1]
        if q1 == q2 and (not y1) and y2:
            cands.append((q1, a1, a2))
            i += 2
        else:
            i += 1
    return select_length_balanced(
        cands, max_pairs=max_pairs, max_ratio=max_ratio, seed=seed
    )


def main() -> None:
    pairs = build_halueval_balanced(max_pairs=300, max_ratio=1.25, seed=42)
    ratios = []
    for i in range(0, len(pairs), 2):
        if i + 1 >= len(pairs):
            break
        ratios.append(length_ratio(pairs[i][1], pairs[i + 1][1]))
    print(
        f"n_pairs_rows={len(pairs)} n_questions≈{len(pairs)//2} "
        f"max_ratio={max(ratios) if ratios else None} "
        f"mean_ratio={sum(ratios)/len(ratios) if ratios else None}"
    )


if __name__ == "__main__":
    main()
