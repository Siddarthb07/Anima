import torch

from core.suppression import detect_suppression


class ToyProbe:
    def heads_from_hidden(self, hidden: torch.Tensor) -> dict[str, float]:
        return {
            "valence": float(hidden[0].item()),
            "arousal": 0.5,
            "uncertainty": float(hidden[1].item()),
        }


def test_detects_valence_shift():
    probe = ToyProbe()
    early = torch.tensor([1.0] + [0.0] * 767)
    late = torch.tensor([2.0] + [0.0] * 767)
    results = [
        {
            "token_text": "x",
            "activations": {0: early, 9: late},
        }
    ]
    events = detect_suppression(results, probe, early_layer=0, late_layer=9)
    assert events and events[0]["suppression_type"] == "valence_suppression"


def test_no_event_when_flat():
    probe = ToyProbe()
    h = torch.zeros(768)
    h[0] = 0.1
    h[1] = 0.5
    results = [{"token_text": "y", "activations": {0: h.clone(), 9: h.clone()}}]
    events = detect_suppression(results, probe, 0, 9)
    assert events == []
