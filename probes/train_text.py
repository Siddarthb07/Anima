"""
Train AffectProbe on GoEmotions (text-native supervision).

Outputs: probes/zoo/{slug}_text.pt + {slug}_text.meta.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from core.extractor import ActivationExtractor
from probes.emotion_va_map import labels_to_va
from probes.linear_probe import AffectProbe
from probes.zoo_io import probe_slug, save_probe_bundle
from core.prompt_format import uses_chat_template


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12
    return float((x * y).sum() / denom)


def _eval_probe(
    probe: AffectProbe,
    acts: List[dict],
    targets: List[dict],
    criterion: nn.Module,
) -> tuple[float, float, float]:
    probe.eval()
    val_loss = 0.0
    pv, gv, pa, ga = [], [], [], []
    with torch.no_grad():
        for i, a in enumerate(acts):
            t = targets[i]
            out = probe(a)
            val_loss += float(
                criterion(out["valence"], torch.tensor(t["valence"]))
                + criterion(out["arousal"], torch.tensor(t["arousal"]))
            )
            pv.append(float(out["valence"]))
            gv.append(t["valence"])
            pa.append(float(out["arousal"]))
            ga.append(t["arousal"])
    val_loss /= max(1, len(acts))
    return (
        val_loss,
        _pearson(np.array(pv), np.array(gv)),
        _pearson(np.array(pa), np.array(ga)),
    )


def build_goemotions_samples(
    extractor: ActivationExtractor,
    *,
    split: str = "train",
    max_samples: int = 2000,
    max_length: int = 128,
) -> Tuple[List[dict], List[dict], dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    acts: List[dict] = []
    targets: List[dict] = []
    import gc

    for i, row in enumerate(ds):
        text = str(row["text"]).strip()
        if len(text) < 3:
            continue
        labels = list(row["labels"])
        v, a, u = labels_to_va(labels)
        rows = extractor.encode_sequence(text[:512], max_length=max_length)
        if not rows:
            continue
        last = rows[-1]
        acts.append(last["activations"])
        targets.append({"valence": v, "arousal": a, "uncertainty": u})
        if (i + 1) % 50 == 0:
            gc.collect()

    meta: dict[str, Any] = {
        "dataset": "go_emotions",
        "split": split,
        "n_samples": len(acts),
        "probe_origin": "text_emotion",
        "use_chat_template": uses_chat_template(extractor.model_name),
    }
    return acts, targets, meta


def train_text_probe(
    model_name: str,
    *,
    max_samples: int = 2000,
    epochs: int = 15,
    device: str = "cpu",
    max_length: int = 128,
    seed: int = 42,
) -> tuple[AffectProbe, dict[str, Any]]:
    extractor = ActivationExtractor(model_name, device=device)
    acts, targets, meta = build_goemotions_samples(
        extractor, max_samples=max_samples, max_length=max_length
    )
    if len(acts) < 32:
        raise RuntimeError(f"Only {len(acts)} GoEmotions samples — increase max_samples or check HF cache.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(acts))
    acts = [acts[i] for i in order]
    targets = [targets[i] for i in order]

    n = len(acts)
    split = int(n * 0.85)
    train_acts, val_acts = acts[:split], acts[split:]
    train_t, val_t = targets[:split], targets[split:]

    probe = AffectProbe(extractor.hidden_dim, len(extractor.layer_indices))
    optimizer = torch.optim.Adam(probe.parameters(), lr=2e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        probe.train()
        total = 0.0
        for i, a in enumerate(train_acts):
            t = train_t[i]
            optimizer.zero_grad()
            out = probe(a)
            loss = (
                criterion(out["valence"], torch.tensor(t["valence"]))
                + criterion(out["arousal"], torch.tensor(t["arousal"]))
                + criterion(out["uncertainty"], torch.tensor(t["uncertainty"]))
            )
            loss.backward()
            optimizer.step()
            total += float(loss.item())

        val_loss, _, _ = _eval_probe(probe, val_acts, val_t, criterion)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1}/{epochs} train_loss={total/len(train_acts):.4f} val_loss={val_loss:.4f}")

    if best_state:
        probe.load_state_dict(best_state)

    _, val_r_v, val_r_a = _eval_probe(probe, val_acts, val_t, criterion)
    meta["val_loss"] = round(best_val, 6)
    meta["val_pearson_valence"] = round(val_r_v, 4)
    meta["val_pearson_arousal"] = round(val_r_a, 4)
    meta["max_length"] = max_length
    meta["train_seed"] = seed
    meta["model"] = model_name
    meta["hidden_dim"] = extractor.hidden_dim
    meta["n_layers"] = len(extractor.layer_indices)
    extractor.cleanup()
    return probe, meta


def main(argv: Optional[List[str]] = None) -> None:
    from core.defaults import DEFAULT_CAUSAL_LM, REFERENCE_CAUSAL_LM_DISTIL

    p = argparse.ArgumentParser(description="Train text-emotion probe (GoEmotions)")
    p.add_argument("--model", default=REFERENCE_CAUSAL_LM_DISTIL)
    p.add_argument("--max-samples", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=12)
    args = p.parse_args(argv)

    model = args.model or DEFAULT_CAUSAL_LM
    probe, meta = train_text_probe(model, max_samples=args.max_samples, epochs=args.epochs)
    slug = probe_slug(model)
    path = save_probe_bundle(probe, slug, meta, suffix="_text")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
