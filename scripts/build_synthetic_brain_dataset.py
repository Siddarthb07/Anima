"""
Build a minimal Narratives-shaped dataset for brain-probe training on low RAM.

Layout matches alignment/narratives_loader.py. BOLD is synthetic (structured noise +
word-rate coupling) — use OpenNeuro ds002345 for real fMRI via scripts/download_narratives_minimal.py.

Output default: data/narratives_minimal/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "narratives_minimal"

STORIES: dict[str, str] = {
    "pieman": (
        "The man walked into the cafe and ordered coffee. He felt nervous about the meeting. "
        "A woman smiled and asked about the weather. The conversation turned serious quickly."
    ),
    "tunnel": (
        "They entered the dark tunnel together. Footsteps echoed on the wet stone. "
        "Fear gave way to relief when light appeared ahead. Everyone laughed with joy."
    ),
    "lucy": (
        "Lucy opened the letter and read it twice. Anger mixed with sadness as she understood. "
        "Later she sat calmly by the window and thought about forgiveness and hope."
    ),
}

SUBJECTS = ["01", "02", "03"]
TR = 2.0
N_VOX = 200  # small spatial grid for fast IO


def _words_from_text(text: str) -> list[dict]:
    words = text.split()
    timings = []
    t = 0.0
    for w in words:
        timings.append({"word": w, "onset_sec": round(t, 3)})
        t += 0.45 + 0.05 * np.random.random()
    return timings


def _synthetic_bold(n_trs: int, word_timings: list[dict], seed: int) -> np.ndarray:
    """Shape (n_voxels, n_trs) before loader transpose -> (n_trs, n_voxels)."""
    rng = np.random.default_rng(seed)
    n_voxels = N_VOX
    bold = rng.standard_normal((n_voxels, n_trs)) * 0.3
    rate = np.zeros(n_trs)
    for wt in word_timings:
        tr_i = int(float(wt["onset_sec"]) / TR)
        if 0 <= tr_i < n_trs:
            rate[tr_i] += 1.0
    for t in range(n_trs):
        bold[:, t] += rate[t] * rng.standard_normal(n_voxels) * 0.5
    # Low-rank emotion-like structure for PCA targets
    v_axis = rng.standard_normal(n_voxels)
    a_axis = rng.standard_normal(n_voxels)
    for t in range(n_trs):
        phase = t / max(1, n_trs - 1)
        bold[:, t] += v_axis * np.sin(phase * np.pi) * 0.4
        bold[:, t] += a_axis * np.cos(phase * np.pi * 0.5) * 0.35
    return bold.astype(np.float32)


def build(out_dir: Path = DEFAULT_OUT) -> Path:
    import nibabel as nib

    out_dir = Path(out_dir)
    stim = out_dir / "stimuli"
    stim.mkdir(parents=True, exist_ok=True)

    for story, text in STORIES.items():
        (stim / f"{story}.txt").write_text(text, encoding="utf-8")
        timings = _words_from_text(text)
        (stim / f"{story}_words.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
        n_trs = int(timings[-1]["onset_sec"] / TR) + 20

        for sub in SUBJECTS:
            subj_seed = hash((story, sub)) % (2**31)
            bold = _synthetic_bold(n_trs, timings, subj_seed)
            # nibabel expects 4D (x,y,z,t); we use 10x10x2 voxels
            nx, ny, nz = 10, 10, 2
            vol = bold.T.reshape(n_trs, nx, ny, nz).transpose(1, 2, 3, 0)
            func_dir = out_dir / f"sub-{sub}" / "func"
            func_dir.mkdir(parents=True, exist_ok=True)
            fname = f"sub-{sub}_task-{story}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
            img = nib.Nifti1Image(vol, affine=np.eye(4))
            nib.save(img, str(func_dir / fname))

    meta = {
        "source": "synthetic_brain_minimal",
        "note": "For pipeline dev; replace with ds002345 for real benchmarks.",
        "stories": list(STORIES.keys()),
        "subjects": SUBJECTS,
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Built synthetic Narratives layout at {out_dir}")
    return out_dir


if __name__ == "__main__":
    build()
