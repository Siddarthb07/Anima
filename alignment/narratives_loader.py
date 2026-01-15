"""
Loads the Narratives fMRI dataset from OpenNeuro.
Dataset: ds002345 — see paper Nastase et al. 2021.
"""

from pathlib import Path


class NarrativesLoader:
    def __init__(self, data_root: str):
        self.root = Path(data_root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Narratives dataset not found at {data_root}. "
                f"Download from: https://openneuro.org/datasets/ds002345"
            )

    def load_story_text(self, story: str) -> str:
        path = self.root / "stimuli" / f"{story}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Stimulus text not found: {path}")
        return path.read_text(encoding="utf-8")

    def load_story_words_with_timing(self, story: str) -> list:
        path = self.root / "stimuli" / f"{story}_words.json"
        if not path.exists():
            raise FileNotFoundError(f"Word timing file not found: {path}")
        import json

        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def load_subject_fmri(self, subject: str, story: str):
        import nibabel as nib

        sub_dir = self.root / f"sub-{subject}" / "func"
        pattern = f"sub-{subject}_task-{story}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        path = sub_dir / pattern
        if not path.exists():
            candidates = sorted(sub_dir.glob(f"sub-{subject}_task-{story}*bold.nii.gz"))
            if not candidates:
                raise FileNotFoundError(f"fMRI file not found for sub-{subject} story {story}")
            path = candidates[0]
        img = nib.load(str(path))
        data = img.get_fdata()
        return data.reshape(-1, data.shape[-1]).T

    def get_available_subjects(self, story: str) -> list:
        subjects = []
        for sub_dir in sorted(self.root.glob("sub-*")):
            func = sub_dir / "func"
            if not func.is_dir():
                continue
            hits = list(func.glob(f"{sub_dir.name}_task-{story}*bold.nii.gz"))
            if hits:
                subjects.append(sub_dir.name.replace("sub-", ""))
        return subjects
