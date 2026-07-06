"""
Steering diagnostic — mutate residual stream along probe directions vs random controls.
"""

import torch


HEDGE_WORDS = {
    "however",
    "although",
    "might",
    "perhaps",
    "possibly",
    "unclear",
    "uncertain",
    "depends",
    "maybe",
    "could",
}


def hedge_score(text: str) -> int:
    return sum(w in text.lower().split() for w in HEDGE_WORDS)


def steer_with_residual_add(model, layer_module, direction: torch.Tensor, alpha: float, forward_fn):
    direction = direction.to(dtype=torch.float16)

    def hook(module, inp, out):
        if isinstance(out, tuple):
            x = out[0]
            delta = alpha * direction.to(x.device)
            x = x + delta.view(1, 1, -1)
            return (x,) + out[1:]
        x = out
        return x + alpha * direction.to(x.device).view(1, 1, -1)

    handle = layer_module.register_forward_hook(hook)
    try:
        return forward_fn()
    finally:
        handle.remove()


def validate_with_controls(extractor, probe, prompt: str, n_controls: int = 5):
    import numpy as np

    from core.hooks import get_decoder_layers

    layers = get_decoder_layers(extractor.model)
    target_layer_idx = extractor.layer_indices[len(extractor.layer_indices) // 2]
    layer_mod = layers[target_layer_idx]

    u_weight = probe.uncertainty.weight.detach().float().squeeze(0)
    direction = u_weight / (u_weight.norm() + 1e-8)

    baseline = extractor.extract(prompt, max_new_tokens=80)
    baseline_text = "".join(r["token_text"] for r in baseline)
    baseline_score = hedge_score(baseline_text)

    def run_steered(vec: torch.Tensor) -> str:
        def fwd():
            return extractor.extract(prompt, max_new_tokens=80)

        return "".join(
            r["token_text"]
            for r in steer_with_residual_add(extractor.model, layer_mod, vec, 20.0, fwd)
        )

    experimental_text = run_steered(direction)
    experimental_score = hedge_score(experimental_text)

    control_scores = []
    for _ in range(n_controls):
        rand_dir = torch.randn_like(direction)
        rand_dir = rand_dir / (rand_dir.norm() + 1e-8)
        control_scores.append(hedge_score(run_steered(rand_dir)))

    return {
        "baseline_score": baseline_score,
        "experimental_score": experimental_score,
        "control_mean": round(float(np.mean(control_scores)), 2),
        "control_max": max(control_scores),
        "signal_is_real": experimental_score > max(control_scores),
        "effect_size": float(experimental_score - np.mean(control_scores)),
    }


def validate_volatility_ablation(extractor, probe, prompt: str, max_new_tokens: int = 48):
    """Compare token-to-token valence std with vs without dampen intervention."""
    import numpy as np

    baseline_rows = extractor.extract(prompt, max_new_tokens=max_new_tokens, intervention_mode="none")
    steered_rows = extractor.extract(
        prompt,
        max_new_tokens=max_new_tokens,
        probe=probe,
        intervention_mode="dampen",
    )

    def _valence_std(rows):
        if len(rows) < 2:
            return 0.0
        vals = [float(probe.predict(r["activations"])["valence"]) for r in rows]
        return float(np.std(vals))

    return {
        "baseline_valence_std": round(_valence_std(baseline_rows), 4),
        "steered_valence_std": round(_valence_std(steered_rows), 4),
        "reduced": _valence_std(steered_rows) < _valence_std(baseline_rows),
        "n_tokens_baseline": len(baseline_rows),
        "n_tokens_steered": len(steered_rows),
    }
