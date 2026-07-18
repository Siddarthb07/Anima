"""Validate a Narratives / ds002345 root against NarrativesLoader expectations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_STORIES = ("pieman", "tunnel", "lucy")


def validate_root(root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "root": str(root.resolve()) if root.exists() else str(root),
        "exists": root.is_dir(),
        "ok": False,
        "issues": [],
        "stories": {},
        "n_subjects": 0,
        "synthetic_marker": None,
        "recommended_probe_origin": None,
    }
    if not root.is_dir():
        report["issues"].append("root does not exist or is not a directory")
        return report

    meta_path = root / "dataset_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            report["synthetic_marker"] = meta.get("source")
            if meta.get("source") == "synthetic_brain_minimal":
                report["recommended_probe_origin"] = "narratives_fMRI_synthetic_minimal"
                report["issues"].append(
                    "dataset_meta.json marks synthetic_brain_minimal — do not claim real fMRI"
                )
            else:
                report["recommended_probe_origin"] = "narratives_fMRI"
        except Exception as exc:
            report["issues"].append(f"could not parse dataset_meta.json: {exc}")
    else:
        report["recommended_probe_origin"] = "narratives_fMRI"

    stimuli = root / "stimuli"
    if not stimuli.is_dir():
        report["issues"].append("missing stimuli/ directory (loader expects stimuli/{story}.txt)")
        # Detect wrong top-level-per-story layout from older docs
        wrong = [s for s in REQUIRED_STORIES if (root / s).is_dir()]
        if wrong:
            report["issues"].append(
                f"found top-level story dirs {wrong} — rearrange to stimuli/ + sub-*/func/ "
                "(see scripts/download_narratives.md)"
            )
        return report

    subjects = sorted(p.name.replace("sub-", "") for p in root.glob("sub-*") if p.is_dir())
    report["n_subjects"] = len(subjects)

    for story in REQUIRED_STORIES:
        info: dict[str, Any] = {
            "text": (stimuli / f"{story}.txt").is_file(),
            "words": (stimuli / f"{story}_words.json").is_file(),
            "subjects_with_bold": [],
        }
        for sub in subjects:
            func = root / f"sub-{sub}" / "func"
            if not func.is_dir():
                continue
            hits = list(func.glob(f"sub-{sub}_task-{story}*bold.nii.gz"))
            if hits:
                info["subjects_with_bold"].append(sub)
        if not info["text"]:
            report["issues"].append(f"missing stimuli/{story}.txt")
        if not info["words"]:
            report["issues"].append(f"missing stimuli/{story}_words.json")
        if not info["subjects_with_bold"]:
            report["issues"].append(f"no BOLD files for story {story}")
        report["stories"][story] = info

    report["ok"] = len(report["issues"]) == 0 or (
        report["recommended_probe_origin"] == "narratives_fMRI_synthetic_minimal"
        and all(report["stories"].get(s, {}).get("text") for s in REQUIRED_STORIES)
        and report["n_subjects"] > 0
        and not any("missing stimuli/" in i or "no BOLD" in i for i in report["issues"] if "synthetic" not in i)
    )
    # ok if layout complete (synthetic warning alone doesn't fail layout)
    layout_ok = (
        all(report["stories"].get(s, {}).get("text") for s in REQUIRED_STORIES)
        and all(report["stories"].get(s, {}).get("words") for s in REQUIRED_STORIES)
        and report["n_subjects"] > 0
        and all(report["stories"].get(s, {}).get("subjects_with_bold") for s in REQUIRED_STORIES)
    )
    report["layout_ok"] = layout_ok
    report["ok"] = layout_ok
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Validate Narratives/ds002345 root for Anima")
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root (default: $NARRATIVES_ROOT or data/narratives_minimal)",
    )
    p.add_argument("--json", action="store_true", help="Print JSON only")
    args = p.parse_args()
    root = args.root
    if root is None:
        import os

        env = os.environ.get("NARRATIVES_ROOT", "").strip()
        root = Path(env) if env else Path(__file__).resolve().parent.parent / "data" / "narratives_minimal"
    report = validate_root(root.expanduser())
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"root: {report['root']}")
        print(f"layout_ok: {report['layout_ok']}  subjects: {report['n_subjects']}")
        print(f"probe_origin: {report['recommended_probe_origin']}")
        if report["synthetic_marker"]:
            print(f"marker: {report['synthetic_marker']}")
        for story, info in report["stories"].items():
            print(
                f"  {story}: text={info['text']} words={info['words']} "
                f"bold_subjects={len(info['subjects_with_bold'])}"
            )
        if report["issues"]:
            print("issues:")
            for i in report["issues"]:
                print(f"  - {i}")
        else:
            print("No issues.")
    raise SystemExit(0 if report.get("layout_ok") else 1)


if __name__ == "__main__":
    main()
