"""
Training probes using Narratives + encoding pipeline utilities.

Targets default to PCA of BOLD on train split (no invented atlas labels).
Optional atlas ROI indices via numpy `.npz` aligned with voxel ordering.
"""

from __future__ import annotations

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
from alignment.word_token_align import word_last_token_indices
from core.extractor import ActivationExtractor
from probes.linear_probe import AffectProbe


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


def build_training_set(
    extractor: ActivationExtractor,
    narratives_root: str,
    stories: list[str],
    *,
    max_subjects_per_story: int = 10,
    target_mode: str = "pca",
    roi_npz_path: Optional[str] = None,
    max_story_chars: int = 8000,
) -> Tuple[List[Dict[int, torch.Tensor]], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    loader = NarrativesLoader(narratives_root)
    pipeline = EncodingAlignmentPipeline(tr=2.0, hrf_lag_trs=3)
    roi_indices = load_roi_index_npz(roi_npz_path) if (roi_npz_path and target_mode == "atlas") else None

    all_activations: List[Dict[int, torch.Tensor]] = []
    Y_rows: list[np.ndarray] = []
    uncertainty_rows: list[float] = []

    metrics: dict[str, Any] = {"stories_used": [], "baselines": {}}

    for story in stories:
        try:
            word_timings = loader.load_story_words_with_timing(story)
            story_text = loader.load_story_text(story)
        except FileNotFoundError:
            continue

        story_text = story_text[:max_story_chars]
        try:
            tok_indices = word_last_token_indices(extractor.tokenizer, story_text, word_timings)
        except Exception:
            continue

        token_rows = extractor.encode_sequence(story_text)
        if len(token_rows) < 8:
            continue

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

        n_trs = mean_fmri.shape[0]
        layer_feats: Dict[int, np.ndarray] = {}
        for layer_idx in extractor.layer_indices:
            layer_feats[layer_idx] = pipeline.align_token_indices_to_tr(
                tok_indices,
                token_rows,
                word_timings,
                n_trs,
                layer_idx,
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

        base_llm = pipeline.word_rate_baseline_predictivity(word_rate, mean_fmri)
        words_only = [str(w.get("word", "")) for w in word_timings[:actual_len]]
        base_lex = pipeline.shallow_lexical_baseline(words_only, mean_fmri)

        metrics["baselines"][story] = {
            "word_rate_mean_r": base_llm["mean_r"],
            "lexical_len_mean_r": base_lex["mean_r"],
        }

        for t in range(actual_len):
            acts = {
                layer_idx: torch.tensor(layer_feats[layer_idx][t], dtype=torch.float32)
                for layer_idx in extractor.layer_indices
            }
            all_activations.append(acts)
            Y_rows.append(fmri_lagged[t])
            uncertainty_rows.append(float(u_tr_lagged[t]))

        metrics["stories_used"].append(story)

    if len(all_activations) < 20:
        raise RuntimeError(
            f"Only {len(all_activations)} training samples — check Narratives path, stories, tokenizer offsets."
        )

    Y = np.stack(Y_rows, axis=0)

    split = int(len(all_activations) * 0.8)
    Y_train = Y[:split]

    if target_mode == "pca":
        tgt = PCATargetTransformer(n_components=2)
        tgt.fit(Y_train)
        valence_all, arousal_all = tgt.transform(Y)
    elif target_mode == "atlas":
        if not roi_indices:
            raise ValueError("target_mode='atlas' requires roi_npz_path")
        valence_list = []
        arousal_list = []
        for t in range(Y.shape[0]):
            means = atlas_roi_means_row(Y[t], roi_indices)
            v, a = roi_dict_to_valence_arousal(means)
            valence_list.append(v)
            arousal_list.append(a)
        valence_all = np.asarray(valence_list, dtype=np.float64)
        arousal_all = np.asarray(arousal_list, dtype=np.float64)
    else:
        raise ValueError("target_mode must be 'pca' or 'atlas'")

    uncertainty_all = np.asarray(uncertainty_rows, dtype=np.float64)

    meta = {
        **metrics,
        "target_mode": target_mode,
        "n_samples": len(all_activations),
    }

    return all_activations, valence_all, arousal_all, uncertainty_all, meta


def train_probe(
    extractor: ActivationExtractor,
    narratives_root: str,
    model_slug: str,
    stories: Optional[List[str]] = None,
    *,
    target_mode: str = "pca",
    roi_npz_path: Optional[str] = None,
) -> tuple[AffectProbe, dict]:
    stories = stories or ["pieman", "tunnel", "lucy"]
    acts, valence, arousal, uncertainty, meta = build_training_set(
        extractor,
        narratives_root,
        stories,
        target_mode=target_mode,
        roi_npz_path=roi_npz_path,
    )

    n_layers = len(extractor.layer_indices)
    probe = AffectProbe(hidden_dim=extractor.hidden_dim, num_layers=n_layers)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    split = int(len(acts) * 0.8)
    train_acts, val_acts = acts[:split], acts[split:]
    train_v, val_v = valence[:split], valence[split:]
    train_a, val_a = arousal[:split], arousal[split:]
    train_u, val_u = uncertainty[:split], uncertainty[split:]

    for epoch in range(30):
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

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/30 — loss: {total_loss:.4f}")

    zoo_dir = Path(__file__).resolve().parent / "zoo"
    zoo_dir.mkdir(parents=True, exist_ok=True)
    out_path = zoo_dir / f"{model_slug}.pt"
    torch.save(probe.state_dict(), out_path)
    print(f"Saved probe: {out_path}")
    meta["checkpoint"] = str(out_path)
    return probe, meta
