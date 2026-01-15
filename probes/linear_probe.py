import torch
import torch.nn as nn
from typing import Dict, Union


class AffectProbe(nn.Module):
    """
    Multi-layer softmax-weighted fusion then three linear heads (valence, arousal, uncertainty).
    """

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.valence = nn.Linear(hidden_dim, 1)
        self.arousal = nn.Linear(hidden_dim, 1)
        self.uncertainty = nn.Linear(hidden_dim, 1)

    def fuse_layers(self, activations: dict[int, torch.Tensor]) -> torch.Tensor:
        if len(activations) != self.num_layers:
            raise ValueError(
                f"AffectProbe expects {self.num_layers} layers, got {len(activations)}"
            )
        stacked = torch.stack(list(activations.values()))
        weights = torch.softmax(self.layer_weights, dim=0)
        return (stacked * weights.unsqueeze(1)).sum(0)

    def forward(self, activations: dict[int, torch.Tensor]) -> dict[str, torch.Tensor]:
        fused = self.fuse_layers(activations)
        fused_b = fused.unsqueeze(0)
        return {
            "valence": torch.tanh(self.valence(fused_b)).squeeze(0).squeeze(-1),
            "arousal": torch.sigmoid(self.arousal(fused_b)).squeeze(0).squeeze(-1),
            "uncertainty": torch.sigmoid(self.uncertainty(fused_b)).squeeze(0).squeeze(-1),
        }

    def heads_from_hidden(self, hidden: torch.Tensor) -> dict[str, float]:
        """Apply heads directly (same weights as fused heads) for layer disagreement metrics."""
        h = hidden.detach().float().reshape(1, -1)
        return {
            "valence": float(torch.tanh(self.valence(h)).squeeze()),
            "arousal": float(torch.sigmoid(self.arousal(h)).squeeze()),
            "uncertainty": float(torch.sigmoid(self.uncertainty(h)).squeeze()),
        }

    def predict(self, activations: Union[Dict[int, torch.Tensor], torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            if isinstance(activations, dict):
                out = self.forward(activations)
            else:
                h = activations.detach().float().reshape(1, -1)
                out = {
                    "valence": torch.tanh(self.valence(h)).squeeze(0).squeeze(-1),
                    "arousal": torch.sigmoid(self.arousal(h)).squeeze(0).squeeze(-1),
                    "uncertainty": torch.sigmoid(self.uncertainty(h)).squeeze(0).squeeze(-1),
                }
        return {
            "valence": round(float(out["valence"].clamp(-1, 1)), 4),
            "arousal": round(float(out["arousal"].clamp(0, 1)), 4),
            "uncertainty": round(float(out["uncertainty"].clamp(0, 1)), 4),
        }
