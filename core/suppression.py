"""
Layer disagreement: compares early-layer vs late-layer readouts using the same linear heads.

Large shifts are heuristic dashboard signals, not claims about subjective experience.
"""

from typing import Any, List, Protocol

VALENCE_SUPPRESSION_THRESHOLD = 0.35
UNCERTAINTY_OVERCLAIM_THRESHOLD = -0.30


class ProbeHeads(Protocol):
    def heads_from_hidden(self, hidden: Any) -> dict:
        ...


def detect_suppression(
    results: list,
    probe: ProbeHeads,
    early_layer: int,
    late_layer: int,
) -> list:
    events = []
    for i, result in enumerate(results):
        early_act = result["activations"].get(early_layer)
        late_act = result["activations"].get(late_layer)
        if early_act is None or late_act is None:
            continue

        early_affect = probe.heads_from_hidden(early_act)
        late_affect = probe.heads_from_hidden(late_act)

        valence_shift = late_affect["valence"] - early_affect["valence"]
        uncertainty_shift = late_affect["uncertainty"] - early_affect["uncertainty"]

        suppression_type = None
        if valence_shift > VALENCE_SUPPRESSION_THRESHOLD:
            suppression_type = "valence_suppression"
        elif uncertainty_shift < UNCERTAINTY_OVERCLAIM_THRESHOLD:
            suppression_type = "uncertainty_overclaim"

        if suppression_type:
            events.append(
                {
                    "token_index": i,
                    "token_text": result.get("token_text", ""),
                    "suppression_type": suppression_type,
                    "early_affect": {
                        "valence": round(float(early_affect["valence"]), 4),
                        "arousal": round(float(early_affect["arousal"]), 4),
                        "uncertainty": round(float(early_affect["uncertainty"]), 4),
                    },
                    "late_affect": {
                        "valence": round(float(late_affect["valence"]), 4),
                        "arousal": round(float(late_affect["arousal"]), 4),
                        "uncertainty": round(float(late_affect["uncertainty"]), 4),
                    },
                    "valence_shift": round(float(valence_shift), 4),
                    "uncertainty_shift": round(float(uncertainty_shift), 4),
                    "severity": "HIGH"
                    if abs(valence_shift) > 0.5 or abs(uncertainty_shift) > 0.45
                    else "MEDIUM",
                }
            )

    return events
