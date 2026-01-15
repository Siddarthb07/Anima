import torch
from typing import Dict, List


def get_decoder_layers(model: torch.nn.Module):
    """Resolve transformer block list for common HF causal LM layouts."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model layout for ActivationHook — add mapping in hooks.py")


class ActivationHook:
    """
    Registers forward hooks on target transformer layers.
    Captures full sequence residual stream activations.
    Clear buffer before each forward pass.
    Call remove() when shutting down the extractor to avoid leaks.
    """

    def __init__(self, model: torch.nn.Module, layer_indices: List[int]):
        self.buffer: Dict[int, torch.Tensor] = {}
        self.handles: List = []
        layers = get_decoder_layers(model)
        for idx in layer_indices:
            layer = layers[idx]

            def make_hook(i: int):
                def _hook(module, inp, out):
                    hidden = out[0] if isinstance(out, tuple) else out
                    self._capture(i, hidden)

                return _hook

            handle = layer.register_forward_hook(make_hook(idx))
            self.handles.append(handle)

    def _capture(self, layer_idx: int, hidden: torch.Tensor):
        self.buffer[layer_idx] = hidden.detach().cpu().float()

    def last_token(self, layer_idx: int) -> torch.Tensor:
        """[hidden_dim] vector for the most recently generated token."""
        t = self.buffer[layer_idx]
        return t[0, -1, :]

    def all_positions(self, layer_idx: int) -> torch.Tensor:
        """[seq_len, hidden_dim] for full sequence analysis."""
        return self.buffer[layer_idx][0]

    def clear(self):
        self.buffer.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
