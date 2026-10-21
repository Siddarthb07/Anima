"""
Push real Narratives fMRI correlations as far as the locked subset allows.

Reports three ceilings (honest, distinct constructs):

1) Word-rate / lexical baselines → mean voxel Pearson r (encoding literature baseline)
2) Anima PCA→VA probe holdout r (what the product currently claims as brain probe)
3) Ridge encoding: LM layer features → top-K PCA BOLD components (deeper encoding path)

Does NOT invent emotion-from-brain SOTA. Writes benchmarks/reports/fmri_depth_*.json.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "benchmarks" / "reports"


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def story_baselines(root: str, story: str, max_subjects: int = 3) -> dict[str, Any]:
    from alignment.encoding_pipeline import EncodingAlignmentPipeline
    from alignment.narratives_loader import NarrativesLoader

    loader = NarrativesLoader(root)
    pipe = EncodingAlignmentPipeline(tr=2.0, hrf_lag_trs=3)
    words = loader.load_story_words_with_timing(story)
    subjects = loader.get_available_subjects(story)[:max_subjects]
    blocks = []
    for s in subjects:
        try:
            blocks.append(loader.load_subject_fmri(s, story))
        except FileNotFoundError:
            continue
    if not blocks:
        return {"status": "missing_bold", "story": story}
    min_t = min(b.shape[0] for b in blocks)
    mean_fmri = np.mean([b[:min_t] for b in blocks], axis=0)
    wr = pipe.compute_word_rate(words, min_t)
    # lag-align like train.py
    wr_l = wr[pipe.hrf_lag_trs :][: min_t - pipe.hrf_lag_trs]
    fmri_l = mean_fmri[pipe.hrf_lag_trs :][: len(wr_l)]
    # subsample voxels for speed if huge
    n_vox = fmri_l.shape[1]
    if n_vox > 8000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_vox, size=8000, replace=False)
        fmri_s = fmri_l[:, idx]
    else:
        fmri_s = fmri_l
        idx = None
    rs = []
    for v in range(fmri_s.shape[1]):
        r = _pearson(wr_l, fmri_s[:, v])
        if r == r:
            rs.append(r)
    rs = np.asarray(rs, dtype=float)
    return {
        "status": "ok",
        "story": story,
        "n_subjects": len(blocks),
        "n_trs": int(len(wr_l)),
        "n_voxels_full": int(n_vox),
        "n_voxels_scored": int(len(rs)),
        "word_rate_mean_r": float(np.nanmean(rs)) if len(rs) else None,
        "word_rate_max_r": float(np.nanmax(rs)) if len(rs) else None,
        "word_rate_p95_r": float(np.nanpercentile(rs, 95)) if len(rs) else None,
        "word_rate_frac_r_gt_0_1": float(np.mean(rs > 0.1)) if len(rs) else None,
        "construct": "stimulus_feature_to_BOLD_encoding_baseline",
    }


def ridge_lm_to_bold_pca(
    model: str,
    root: str,
    train_stories: list[str],
    holdout_stories: list[str],
    *,
    n_components: int = 20,
    max_subjects: int = 3,
    max_chars: int = 6000,
) -> dict[str, Any]:
    """Deeper path: predict BOLD PCA scores from LM TR features (not VA labels)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV

    from alignment.encoding_pipeline import EncodingAlignmentPipeline
    from alignment.narratives_loader import NarrativesLoader
    from core.extractor import ActivationExtractor

    loader = NarrativesLoader(root)
    pipe = EncodingAlignmentPipeline(tr=2.0, hrf_lag_trs=3)
    ex = ActivationExtractor(model)

    def story_xy(story: str):
        words = loader.load_story_words_with_timing(story)
        text = loader.load_story_text(story)[:max_chars]
        subjects = loader.get_available_subjects(story)[:max_subjects]
        blocks = []
        for s in subjects:
            try:
                blocks.append(loader.load_subject_fmri(s, story))
            except FileNotFoundError:
                continue
        if not blocks:
            return None
        min_t = min(b.shape[0] for b in blocks)
        mean_fmri = np.mean([b[:min_t] for b in blocks], axis=0)
        from alignment.word_token_align import word_last_token_indices

        token_rows = ex.encode_sequence(text)
        tok_indices = word_last_token_indices(ex.tokenizer, text, words)
        layer = ex.layer_indices[len(ex.layer_indices) // 2]
        feats = pipe.align_token_indices_to_tr(tok_indices, token_rows, words, min_t, layer)
        feats_l, fmri_l = pipe.apply_hrf_lag(feats, mean_fmri)
        return feats_l, fmri_l

    Xtr, Ytr = [], []
    for st in train_stories:
        out = story_xy(st)
        if out is None:
            continue
        x, y = out
        Xtr.append(x)
        Ytr.append(y)
    if not Xtr:
        ex.cleanup()
        return {"status": "no_train_data"}
    X_train = np.concatenate(Xtr, axis=0)
    Y_train = np.concatenate(Ytr, axis=0)
    # voxel subsample then PCA
    rng = np.random.default_rng(42)
    if Y_train.shape[1] > 10000:
        vidx = rng.choice(Y_train.shape[1], size=10000, replace=False)
        Y_train = Y_train[:, vidx]
    else:
        vidx = None
    pca = PCA(n_components=min(n_components, Y_train.shape[0] - 1, Y_train.shape[1]))
    Ypc_train = pca.fit_transform(Y_train)
    model_r = RidgeCV(alphas=np.array([0.1, 1.0, 10.0, 100.0, 1000.0]))
    model_r.fit(X_train, Ypc_train)

    hold = {}
    for st in holdout_stories:
        out = story_xy(st)
        if out is None:
            hold[st] = {"status": "missing"}
            continue
        x, y = out
        if vidx is not None:
            y = y[:, vidx]
        ypc = pca.transform(y)
        pred = model_r.predict(x)
        rs = [_pearson(pred[:, k], ypc[:, k]) for k in range(ypc.shape[1])]
        hold[st] = {
            "status": "ok",
            "n_trs": int(x.shape[0]),
            "mean_r_across_pcs": float(np.nanmean(rs)),
            "max_r_across_pcs": float(np.nanmax(rs)),
            "pc_rs": [round(float(r), 4) if r == r else None for r in rs],
        }
    ex.cleanup()
    return {
        "status": "ok",
        "construct": "LM_features_to_BOLD_PCA_encoding",
        "model": model,
        "n_components": int(pca.n_components_),
        "explained_var_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "train_stories": train_stories,
        "holdout": hold,
        "honesty": {
            "not_valence_arousal_labels": True,
            "not_voxel_wise_whole_brain_SOTA": True,
            "note": "Predicts low-D BOLD PCA scores from mid-layer LM TR features on held-out story.",
        },
    }


def anima_va_holdout(
    model: str,
    root: str,
    train_stories: Optional[list[str]] = None,
    holdout_stories: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Current product path: train PCA→VA probe, report holdout Pearson."""
    from core.extractor import ActivationExtractor
    from probes.train import train_probe
    from probes.zoo_io import probe_slug

    ex = ActivationExtractor(model)
    slug = probe_slug(model)
    try:
        _, meta = train_probe(
            ex,
            narratives_root=root,
            model_slug=slug,
            train_stories=train_stories,
            holdout_stories=holdout_stories,
            target_mode="pca",
            epochs=12,
        )
    finally:
        ex.cleanup()
    return {
        "status": "ok",
        "construct": "Anima_PCA_BOLD_to_valence_arousal_probe",
        "model": model,
        "slug": slug,
        "val_r_valence": meta.get("val_r_valence"),
        "val_r_arousal": meta.get("val_r_arousal"),
        "val_mse": meta.get("val_mse"),
        "baselines": meta.get("baselines"),
        "probe_origin": meta.get("probe_origin"),
        "n_train": meta.get("n_train"),
        "n_val": meta.get("n_val"),
        "honesty": {
            "PCA_components_are_not_emotion_labels": True,
            "holdout_is_story_lucy": True,
        },
    }


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("NARRATIVES_ROOT", str(ROOT / "data" / "ds002345")))
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-encoding", action="store_true")
    p.add_argument(
        "--split",
        default="",
        help="Optional holdout JSON (default: narratives_holdout.json or depth subset)",
    )
    args = p.parse_args(argv)

    root = args.root
    depth_split = ROOT / "benchmarks" / "splits" / "narratives_holdout_depth.json"
    split_path = Path(args.split) if args.split else (
        depth_split if depth_split.is_file() else ROOT / "benchmarks" / "splits" / "narratives_holdout.json"
    )
    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.is_file() else {
        "train": ["pieman", "tunnel"],
        "holdout": ["lucy"],
    }
    train_stories = list(split.get("train") or ["pieman"])
    holdout_stories = list(split.get("holdout") or ["lucy"])
    stories = list(dict.fromkeys([*train_stories, *holdout_stories]))

    report: dict[str, Any] = {
        "schema": "anima.fmri_depth.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "narratives_root": root,
        "model": args.model,
        "split_path": str(split_path),
        "train_stories": train_stories,
        "holdout_stories": holdout_stories,
        "stories": {},
        "anima_va_probe": None,
        "ridge_encoding": None,
    }

    try:
        import subprocess

        proc = subprocess.run(
            ["python", str(ROOT / "scripts" / "validate_narratives_root.py"), "--root", root],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        report["validate_stdout"] = (proc.stdout or "")[-2000:]
        report["validate_ok"] = proc.returncode == 0
    except Exception as exc:
        report["validate_error"] = str(exc)

    for story in stories:
        print(f"[baseline] {story}", flush=True)
        report["stories"][story] = story_baselines(root, story)

    if not args.skip_train:
        print("[anima VA probe] training…", flush=True)
        try:
            report["anima_va_probe"] = anima_va_holdout(args.model, root, train_stories, holdout_stories)
        except Exception as exc:
            report["anima_va_probe"] = {"status": "error", "message": str(exc)}

    if not args.skip_encoding:
        print("[ridge LM→BOLD PCA] fitting…", flush=True)
        try:
            report["ridge_encoding"] = ridge_lm_to_bold_pca(
                args.model,
                root,
                train_stories,
                holdout_stories,
            )
        except Exception as exc:
            report["ridge_encoding"] = {"status": "error", "message": str(exc)}

    REPORTS.mkdir(parents=True, exist_ok=True)
    dest = REPORTS / f"fmri_depth_{args.model.replace('/', '_')}.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {dest}", flush=True)
    # short print
    for st, b in report["stories"].items():
        if b.get("status") == "ok":
            print(
                f"  {st}: word_rate mean_r={b['word_rate_mean_r']:.4f} "
                f"max={b['word_rate_max_r']:.4f} p95={b['word_rate_p95_r']:.4f}",
                flush=True,
            )
    va = report.get("anima_va_probe") or {}
    if va.get("status") == "ok":
        print(
            f"  VA probe: r_v={va.get('val_r_valence')} r_a={va.get('val_r_arousal')} "
            f"origin={va.get('probe_origin')}",
            flush=True,
        )
    enc = report.get("ridge_encoding") or {}
    if enc.get("status") == "ok":
        h = (enc.get("holdout") or {}).get("lucy") or {}
        print(
            f"  ridge holdout lucy: mean_r={h.get('mean_r_across_pcs')} max_r={h.get('max_r_across_pcs')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
