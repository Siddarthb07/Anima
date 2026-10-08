"""Semantic-entropy helpers (Kuhn-style clustering) shared by baselines/ablations."""

from __future__ import annotations

import math
import re
from typing import Sequence


def normalize_answer(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cluster_entropy(texts: Sequence[str], *, sim_threshold: float = 0.5) -> float:
    norms = [normalize_answer(t) for t in texts]
    clusters: list[list[str]] = []
    for n in norms:
        placed = False
        for c in clusters:
            if token_jaccard(n, c[0]) >= sim_threshold:
                c.append(n)
                placed = True
                break
        if not placed:
            clusters.append([n])
    total = max(1, sum(len(c) for c in clusters))
    ent = 0.0
    for c in clusters:
        p = len(c) / total
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    max_ent = math.log(max(2, len(texts)))
    return round(ent / max_ent, 4) if max_ent > 0 else 0.0


def claim_disagreement(generations: Sequence[str], answer: str, *, sim_threshold: float = 0.5) -> float:
    """1 − fraction of free-gens that match the given claim (claim-specific uncertainty)."""
    if not generations:
        return 0.5
    target = normalize_answer(answer)
    matches = sum(1 for g in generations if token_jaccard(normalize_answer(g), target) >= sim_threshold)
    return round(1.0 - matches / len(generations), 4)
