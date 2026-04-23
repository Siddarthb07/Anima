"""
Train text + brain (Narratives) probes on the tiny default model (fits constrained RAM).

  python scripts/download_narratives_minimal.py
  python scripts/train_all_probes.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ANIMA_FORCE_CPU", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.defaults import DEFAULT_CAUSAL_LM
from core.extractor import ActivationExtractor
from probes.train import train_probe
from probes.train_text import train_text_probe
from probes.zoo_io import probe_slug, save_probe_bundle


def _narratives_root() -> Path:
    env = os.environ.get("NARRATIVES_ROOT", "")
    if env and Path(env).is_dir():
        return Path(env)
    synth = ROOT / "data" / "narratives_minimal"
    if synth.is_dir() and (synth / "stimuli").exists():
        return synth
    raise FileNotFoundError(
        "No Narratives data. Run: python scripts/download_narratives_minimal.py"
    )


def main() -> None:
    model = DEFAULT_CAUSAL_LM
    slug = probe_slug(model)
    narr_root = _narratives_root()
    os.environ["NARRATIVES_ROOT"] = str(narr_root)
    print(f"Model: {model}")
    print(f"NARRATIVES_ROOT: {narr_root}")

    print("\n--- text probe (GoEmotions) ---")
    probe_t, meta_t = train_text_probe(model, max_samples=50, epochs=6, device="cpu")
    p_text = save_probe_bundle(probe_t, slug, meta_t, suffix="_text")
    print("Saved", p_text)

    print("\n--- brain probe (Narratives PCA) ---")
    ex = ActivationExtractor(model, device="cpu")
    _, meta_b = train_probe(
        ex,
        str(narr_root),
        slug,
        epochs=12,
        target_mode="pca",
    )
    print("Brain meta:", {k: meta_b.get(k) for k in ("probe_origin", "val_r_valence", "val_r_arousal", "checkpoint")})
    ex.cleanup()
    print("\nDone.")


if __name__ == "__main__":
    main()
