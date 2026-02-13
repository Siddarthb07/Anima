"""Map GoEmotions label indices to valence / arousal proxies (Russell-style sketch)."""

from __future__ import annotations

# GoEmotions simplified 28 labels index order (0..27); 27 often neutral.
# Values are rough circumplex coordinates for probe supervision — not clinical norms.
LABEL_VA: dict[int, tuple[float, float]] = {
    0: (0.6, 0.5),   # admiration
    1: (0.7, 0.7),   # amusement
    2: (-0.7, 0.8),  # anger
    3: (0.5, 0.3),   # approval
    4: (0.6, 0.4),   # caring
    5: (-0.2, 0.3),  # confusion
    6: (0.3, 0.5),   # curiosity
    7: (0.5, 0.6),   # desire
    8: (-0.6, 0.2),  # disappointment
    9: (-0.7, 0.5),  # disapproval
    10: (-0.8, 0.6), # disgust
    11: (0.8, 0.8),  # excitement
    12: (-0.8, 0.9), # fear
    13: (0.7, 0.4),  # gratitude
    14: (-0.9, 0.2), # grief
    15: (0.8, 0.7),  # joy
    16: (0.4, 0.5),  # love
    17: (-0.5, 0.4), # nervousness
    18: (0.2, 0.2),  # optimism
    19: (0.5, 0.5),  # pride
    20: (-0.6, 0.7), # remorse
    21: (-0.4, 0.6), # sadness
    22: (-0.3, 0.7), # surprise
    23: (0.0, 0.1),  # neutral
    24: (0.3, 0.4),
    25: (0.2, 0.3),
    26: (0.1, 0.2),
    27: (0.0, 0.0),
}


def labels_to_va(labels: list[int]) -> tuple[float, float, float]:
    """Multi-label -> mean valence, arousal, uncertainty (more labels -> higher u)."""
    if not labels:
        return 0.0, 0.5, 0.55
    vs, ars = [], []
    for li in labels:
        v, a = LABEL_VA.get(int(li), (0.0, 0.5))
        vs.append(v)
        ars.append(a)
    valence = float(sum(vs) / len(vs))
    arousal = float(sum(ars) / len(ars))
    uncertainty = min(0.95, 0.45 + 0.08 * max(0, len(labels) - 1))
    return valence, arousal, uncertainty
