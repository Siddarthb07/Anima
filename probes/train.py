"""
Training probes using Narratives + encoding pipeline utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from alignment.confound_control import build_confound_matrix, residualize
from alignment.encoding_pipeline import EncodingAlignmentPipeline
from alignment.narratives_loader import NarrativesLoader
from alignment.target_builders import (
    PCATargetTransformer,
    atlas_roi_means_row,
    load_roi_index_npz,
    roi_dict_to_valence_arousal,
)
from alignment.tribe_encoder import ROI_DEFINITIONS, TRIBEv2Encoder
from alignment.word_token_align import word_last_token_indices
from core.extractor import ActivationExtractor
from probes.calibration import PlattScaler
from probes.linear_probe import AffectProbe
from probes.zoo_io import probe_slug, save_meta, save_probe_bundle, tribe_weights_path

DEFAULT_HOLDOUT = ["lucy"]
DEFAULT_TRAIN_STORIES = ["pieman", "tunnel"]
SPLIT_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "splits" / "narratives_holdout.json"


def load_narratives_split() -> Tuple[List[str], List[str]]:
    """Leakage-safe train/holdout stories from benchmarks/splits/narratives_holdout.json."""
    if SPLIT_PATH.is_file():
        spec = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        train = spec.get("train") or DEFAULT_TRAIN_STORIES
        holdout = spec.get("holdout") or DEFAULT_HOLDOUT
        return list(train), list(holdout)
    return list(DEFAULT_TRAIN_STORIES), list(DEFAULT_HOLDOUT)


def _uncertainty_by_tr(
    word_last_tokens: list[int],
    token_rows: list[dict],
    word_timings: list[dict],
    n_trs: int,
    tr: float,
) -> np.ndarray:
    fused = np.zeros(n_trs, dtype=np.float64)
    counts = np.zeros(n_trs, dtype=np.float64)
    n_words = min(len(word_timings), len(word_last_tokens))
    for wi in range(n_words):
        ti = int(float(word_timings[wi]["onset_sec"]) / tr)
        tok_i = word_last_tokens[wi]
        if ti < 0 or ti >= n_trs:
            continue
        if tok_i < 0 or tok_i >= len(token_rows):
            continue
        u = float(token_rows[tok_i]["uncertainty_signals"]["fused"])
        fused[ti] += u
        counts[ti] += 1.0
    mask = counts > 0
    fused[mask] /= counts[mask]
    fused[~mask] = 0.5
    return fused


def _story_samples(
    extractor: ActivationExtractor,
    loader: NarrativesLoader,
    pipeline: EncodingAlignmentPipeline,
    story: str,
    *,
    target_mode: str,
    roi_indices: Optional[dict],
    max_subjects: int,
    max_story_chars: int,
    pca_fit: Optional[PCATargetTransformer],
) -> Tuple[list, list, list, list, dict]:
    word_timings = loader.load_story_words_with_timing(story)
    story_text = loader.load_story_text(story)[:max_story_chars]
    tok_indices = word_last_token_indices(extractor.tokenizer, story_text, word_timings)
    token_rows = extractor.encode_sequence(story_text)
    if len(token_rows) < 8:
        return [], [], [], [], {}

    subjects = loader.get_available_subjects(story)[:max_subjects]
    fmri_blocks = []
    for subject in subjects:
        try:
            fmri_blocks.append(loader.load_subject_fmri(subject, story))
        except FileNotFoundError:
            continue
    if not fmri_blocks:
        return [], [], [], [], {}

    min_t = min(f.shape[0] for f in fmri_blocks)
    mean_fmri = np.mean([f[:min_t] for f in fmri_blocks], axis=0)
    n_trs = mean_fmri.shape[0]
    layer_feats: Dict[int, np.ndarray] = {}
    for layer_idx in extractor.layer_indices:
        layer_feats[layer_idx] = pipeline.align_token_indices_to_tr(
            tok_indices, token_rows, word_timings, n_trs, layer_idx
        )
    word_rate = pipeline.compute_word_rate(word_timings, n_trs)
    confounds = build_confound_matrix(word_rate, n_trs)
    for layer_idx in extractor.layer_indices:
        layer_feats[layer_idx] = residualize(layer_feats[layer_idx], confounds)

    first_layer = extractor.layer_indices[0]
    feat_lagged, fmri_lagged = pipeline.apply_hrf_lag(layer_feats[first_layer], mean_fmri)
    actual_len = len(feat_lagged)
    u_tr = _uncertainty_by_tr(tok_indices, token_rows, word_timings, n_trs, pipeline.tr)
    u_tr_lagged = u_tr[pipeline.hrf_lag_trs :][:actual_len]

    baselines = {
        "word_rate_mean_r": pipeline.word_rate_baseline_predictivity(word_rate, mean_fmri)["mean_r"],
        "lexical_len_mean_r": pipeline.shallow_lexical_baseline(
            [str(w.get("word", "")) for w in word_timings[:actual_len]], mean_fmri
        )["mean_r"],
    }

    acts, Y_rows, fused_u = [], [], []
    for t in range(actual_len):
        acts.append(
            {
                layer_idx: torch.tensor(layer_feats[layer_idx][t], dtype=torch.float32)
                for layer_idx in extractor.layer_indices
            }
        )
        Y_rows.append(fmri_lagged[t])
        fused_u.append(float(u_tr_lagged[t]))

    Y = np.stack(Y_rows, axis=0)
    if target_mode == "pca":
        if pca_fit is None:
            raise ValueError("pca_fit required")
        valence, arousal = pca_fit.transform(Y)
    elif target_mode == "atlas":
        if not roi_indices:
            raise ValueError("roi_indices required for atlas")
        valence, arousal = [], []
        for t in range(Y.shape[0]):
            means = atlas_roi_means_row(Y[t], roi_indices)
            v, a = roi_dict_to_valence_arousal(means)
            valence.append(v)
            arousal.append(a)
        valence = np.asarray(valence, dtype=np.float64)
        arousal = np.asarray(arousal, dtype=np.float64)
    else:
        raise ValueError(f"unsupported target_mode {target_mode}")

    return acts, list(valence), list(arousal), fused_u, baselines


def build_training_set(
    extractor: ActivationExtractor,
    narratives_root: str,
    train_stories: list[str],
    holdout_stories: list[str],
    *,
    max_subjects_per_story: int = 10,
    target_mode: str = "pca",
    roi_npz_path: Optional[str] = None,
    max_story_chars: int = 8000,
) -> Tuple[list, list, list, list, list, list, list, list, dict]:
    loader = NarrativesLoader(narratives_root)
    pipeline = EncodingAlignmentPipeline(tr=2.0, hrf_lag_trs=3)
    roi_indices = load_roi_index_npz(roi_npz_path) if (roi_npz_path and target_mode == "atlas") else None

    train_acts, train_v, train_a, train_u = [], [], [], []
    val_acts, val_v, val_a, val_u = [], [], [], []
    metrics: dict[str, Any] = {
        "baselines": {},
        "train_stories": list(train_stories),
        "holdout_stories": list(holdout_stories),
    }

    # Fit PCA on train stories only (BOLD rows from train split)
    Y_train_fit = []
    for story in train_stories:
        try:
            word_timings = loader.load_story_words_with_timing(story)
            story_text = loader.load_story_text(story)[:max_story_chars]
            subjects = loader.get_available_subjects(story)[:max_subjects_per_story]
            fmri_blocks = []
            for subject in subjects:
                try:
                    fmri_blocks.append(loader.load_subject_fmri(subject, story))
                except FileNotFoundError:
                    continue
            if not fmri_blocks:
                continue
            min_t = min(f.shape[0] for f in fmri_blocks)
            mean_fmri = np.mean([f[:min_t] for f in fmri_blocks], axis=0)
            fmri_lagged = mean_fmri[pipeline.hrf_lag_trs :]
            for row in fmri_lagged:
                Y_train_fit.append(row)
        except Exception:
            continue

    pca_fit = None
    if target_mode == "pca" and Y_train_fit:
        pca_fit = PCATargetTransformer(n_components=2)
        pca_fit.fit(np.stack(Y_train_fit, axis=0))

    for story in train_stories:
        acts, v, a, u, base = _story_samples(
            extractor,
            loader,
            pipeline,
            story,
            target_mode=target_mode,
            roi_indices=roi_indices,
            max_subjects=max_subjects_per_story,
            max_story_chars=max_story_chars,
            pca_fit=pca_fit,
        )
        train_acts.extend(acts)
        train_v.extend(v)
        train_a.extend(a)
        train_u.extend(u)
        metrics["baselines"][story] = base

    for story in holdout_stories:
        acts, v, a, u, base = _story_samples(
            extractor,
            loader,
            pipeline,
            story,
            target_mode=target_mode,
            roi_indices=roi_indices,
            max_subjects=max_subjects_per_story,
            max_story_chars=max_story_chars,
            pca_fit=pca_fit,
        )
        val_acts.extend(acts)
        val_v.extend(v)
        val_a.extend(a)
        val_u.extend(u)
        metrics["holdout_stories"].append(story)
        metrics["baselines"][f"holdout_{story}"] = base

    if len(train_acts) < 20:
        raise RuntimeError(f"Only {len(train_acts)} train samples — check Narratives path.")

    origin = "narratives_fMRI"
    meta_path = Path(narratives_root) / "dataset_meta.json"
    if meta_path.exists():
        import json

        dm = json.loads(meta_path.read_text(encoding="utf-8"))
        if dm.get("source") == "synthetic_brain_minimal":
            origin = "narratives_fMRI_synthetic_minimal"
    meta = {
        **metrics,
        "target_mode": target_mode,
        "n_train": len(train_acts),
        "n_val": len(val_acts),
        "probe_origin": origin,
        "narratives_root": str(narratives_root),
    }
    return (
        train_acts,
        train_v,
        train_a,
        train_u,
        val_acts,
        val_v,
        val_a,
        val_u,
        meta,
    )


def _eval_probe(
    probe: AffectProbe,
    acts: list,
    valence: list,
    arousal: list,
) -> dict[str, float]:
    probe.eval()
    pv, gv, pa, ga = [], [], [], []
    mse = 0.0
    crit = nn.MSELoss()
    with torch.no_grad():
        for i, a in enumerate(acts):
            out = probe(a)
            t_v, t_a = float(valence[i]), float(arousal[i])
            mse += float(crit(out["valence"], torch.tensor(t_v)) + crit(out["arousal"], torch.tensor(t_a)))
            pv.append(float(out["valence"]))
            gv.append(t_v)
            pa.append(float(out["arousal"]))
            ga.append(t_a)
    n = max(1, len(acts))
    def _r(x, y):
        x, y = np.array(x), np.array(y)
        if len(x) < 3:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    return {
        "val_mse": mse / n,
        "val_r_valence": _r(pv, gv),
        "val_r_arousal": _r(pa, ga),
    }


def _fit_platt(train_u: list, train_fused: list, val_u: list, val_fused: list) -> PlattScaler:
    cal = PlattScaler()
    raw = torch.tensor(train_fused, dtype=torch.float32)
    targets = torch.tensor([(u + f) / 2 > 0.6 for u, f in zip(train_u, train_fused)], dtype=torch.float32)
    cal.fit(raw, targets)
    return cal


def train_tribe_projection(
    acts: list,
    valence: list,
    arousal: list,
    hidden_dim: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Ridge-style ROI axes predicting valence/arousal proxies from mean hidden."""
    rng = np.random.default_rng(seed)
    weights = {}
    X = []
    for a in acts:
        stacked = torch.stack(list(a.values())).mean(0).numpy()
        X.append(stacked)
    X = np.stack(X, axis=0)
    y_v = np.array(valence, dtype=np.float64)
    y_a = np.array(arousal, dtype=np.float64)
    for roi, target in zip(ROI_DEFINITIONS.keys(), [y_v, y_a, y_v - y_a, y_v + y_a, np.linalg.norm(X, axis=1)]):
        if len(target) != len(X):
            target = y_v
        w, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
        w = w.astype(np.float64)
        w /= np.linalg.norm(w) + 1e-12
        if w.shape[0] != hidden_dim:
            w = rng.standard_normal(hidden_dim)
            w /= np.linalg.norm(w) + 1e-12
        weights[roi] = w
    return weights


def train_probe(
    extractor: ActivationExtractor,
    narratives_root: str,
    model_slug: str,
    train_stories: Optional[List[str]] = None,
    holdout_stories: Optional[List[str]] = None,
    *,
    target_mode: str = "pca",
    roi_npz_path: Optional[str] = None,
    epochs: int = 30,
) -> tuple[AffectProbe, dict]:
    if train_stories is None or holdout_stories is None:
        split_train, split_hold = load_narratives_split()
        train_stories = train_stories or split_train
        holdout_stories = holdout_stories or split_hold
    (
        train_acts,
        train_v,
        train_a,
        train_u,
        val_acts,
        val_v,
        val_a,
        val_u,
        meta,
    ) = build_training_set(
        extractor,
        narratives_root,
        train_stories,
        holdout_stories,
        target_mode=target_mode,
        roi_npz_path=roi_npz_path,
    )

    n_layers = len(extractor.layer_indices)
    probe = AffectProbe(extractor.hidden_dim, n_layers)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        probe.train()
        total_loss = 0.0
        for i in range(0, len(train_acts), 32):
            batch_acts = train_acts[i : i + 32]
            batch_v = torch.tensor(train_v[i : i + 32], dtype=torch.float32)
            batch_a = torch.tensor(train_a[i : i + 32], dtype=torch.float32)
            batch_u = torch.tensor(train_u[i : i + 32], dtype=torch.float32)
            optimizer.zero_grad()
            outs_v, outs_a, outs_u = [], [], []
            for a in batch_acts:
                out = probe(a)
                outs_v.append(out["valence"])
                outs_a.append(out["arousal"])
                outs_u.append(out["uncertainty"])
            preds_v = torch.stack(outs_v).squeeze(-1)
            preds_a = torch.stack(outs_a).squeeze(-1)
            preds_u = torch.stack(outs_u).squeeze(-1)
            loss = criterion(preds_v, batch_v) + criterion(preds_a, batch_a) + criterion(preds_u, batch_u)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        metrics = _eval_probe(probe, val_acts, val_v, val_a)
        if metrics["val_mse"] < best_val:
            best_val = metrics["val_mse"]
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} loss={total_loss:.4f} val_mse={metrics['val_mse']:.4f}")

    if best_state:
        probe.load_state_dict(best_state)

    meta.update(_eval_probe(probe, val_acts, val_v, val_a))
    meta["model"] = extractor.model_name
    suffix = "_narratives_pca" if target_mode == "pca" else f"_narratives_{target_mode}"
    path = save_probe_bundle(probe, model_slug, meta, suffix=suffix)

    if val_acts and train_acts:
        cal = _fit_platt(train_u, train_u, val_u, val_u)
        torch.save(cal.state_dict(), Path(path).parent / f"{model_slug}{suffix}.calib.pt")

    tribe_w = train_tribe_projection(train_acts, train_v, train_a, extractor.hidden_dim, seed=42)
    TRIBEv2Encoder.save_weights(tribe_weights_path(model_slug), tribe_w)

    meta["checkpoint"] = str(path)
    return probe, meta
