"""Train a minimal probe on synthetic targets for tiny-random-gpt2 (fast, no HF datasets)."""

from __future__ import annotations

import os

# Avoid loading CUDA DLLs on memory-constrained Windows hosts when only CPU is needed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

from core.defaults import DEFAULT_CAUSAL_LM
from core.extractor import ActivationExtractor
from probes.linear_probe import AffectProbe
from probes.zoo_io import probe_slug, save_probe_bundle


def main() -> None:
    model = DEFAULT_CAUSAL_LM
    ex = ActivationExtractor(model)
    texts = [
        "I am furious and terrified",
        "Thank you so much, this is wonderful",
        "Whatever.",
        "I love this and feel calm",
        "This is disgusting and awful",
    ] * 8

    acts = []
    valence, arousal, uncertainty = [], [], []
    for t in texts:
        rows = ex.extract(t, max_new_tokens=4)
        if not rows:
            continue
        last = rows[-1]
        acts.append(last["activations"])
        tl = t.lower()
        if "furious" in tl or "disgust" in tl or "awful" in tl:
            valence.append(-0.7)
            arousal.append(0.75)
            uncertainty.append(0.5)
        elif "wonderful" in tl or "love" in tl or "thank" in tl:
            valence.append(0.75)
            arousal.append(0.55)
            uncertainty.append(0.35)
        else:
            valence.append(0.0)
            arousal.append(0.45)
            uncertainty.append(0.55)

    probe = AffectProbe(ex.hidden_dim, len(ex.layer_indices))
    opt = torch.optim.Adam(probe.parameters(), lr=5e-3)
    crit = nn.MSELoss()
    for _ in range(40):
        probe.train()
        for i, a in enumerate(acts):
            opt.zero_grad()
            o = probe(a)
            loss = (
                crit(o["valence"], torch.tensor(valence[i]))
                + crit(o["arousal"], torch.tensor(arousal[i]))
                + crit(o["uncertainty"], torch.tensor(uncertainty[i]))
            )
            loss.backward()
            opt.step()

    slug = probe_slug(model)
    meta = {
        "probe_origin": "synthetic_tiny",
        "model": model,
        "n_samples": len(acts),
        "note": "Quick zoo checkpoint for default tiny model; train GoEmotions/Narratives for real use.",
    }
    path = save_probe_bundle(probe, slug, meta, suffix="")
    print("Wrote", path)
    ex.cleanup()


if __name__ == "__main__":
    main()
