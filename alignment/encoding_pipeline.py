"""
Encoding-model utilities aligned with modern neural-encoding practice.

This module implements TR binning, HRF lag trimming, ridge fitting, and baseline
comparisons using NumPy + scikit-learn. It is **not** a thin wrapper around the
third-party LITcoder library; methodology is informed by that literature.
"""

from typing import Dict, List, Optional

import numpy as np


class EncodingAlignmentPipeline:
    """
    Temporal alignment of LLM features with fMRI TRs, hemodynamic lag handling,
    confound-aware preprocessing hooks, and ridge encoding with baselines.
    """

    def __init__(self, tr: float = 2.0, hrf_lag_trs: int = 3):
        self.tr = tr
        self.hrf_lag_trs = hrf_lag_trs

    def align_token_indices_to_tr(
        self,
        word_last_token_indices: List[int],
        token_rows: List[dict],
        word_timings: List[dict],
        n_timepoints: int,
        layer_idx: int,
    ) -> np.ndarray:
        """
        Average hidden states for tokens belonging to words whose onset falls in each TR bin.

        word_last_token_indices[k] = tokenizer index of last subword token for word k.
        """
        if not token_rows:
            raise ValueError("token_rows empty")

        hidden_dim = token_rows[0]["activations"][layer_idx].shape[-1]
        tr_features = np.zeros((n_timepoints, hidden_dim), dtype=np.float64)
        tr_counts = np.zeros(n_timepoints, dtype=np.float64)

        n_words = min(len(word_timings), len(word_last_token_indices))
        for wi in range(n_words):
            onset_sec = float(word_timings[wi]["onset_sec"])
            tr_idx = int(onset_sec / self.tr)
            if tr_idx >= n_timepoints:
                continue
            tok_idx = word_last_token_indices[wi]
            if tok_idx < 0 or tok_idx >= len(token_rows):
                continue
            act = token_rows[tok_idx]["activations"][layer_idx]
            if hasattr(act, "detach"):
                vec = act.detach().cpu().numpy().astype(np.float64)
            else:
                vec = np.asarray(act, dtype=np.float64)
            tr_features[tr_idx] += vec
            tr_counts[tr_idx] += 1.0

        mask = tr_counts > 0
        tr_features[mask] /= tr_counts[mask, np.newaxis]
        return tr_features

    def apply_hrf_lag(self, features: np.ndarray, fmri: np.ndarray):
        lag = self.hrf_lag_trs
        features_aligned = features[:-lag] if lag > 0 else features
        fmri_aligned = fmri[lag:] if lag > 0 else fmri
        min_len = min(len(features_aligned), len(fmri_aligned))
        return features_aligned[:min_len], fmri_aligned[:min_len]

    def compute_word_rate(self, word_timings: List[dict], n_timepoints: int) -> np.ndarray:
        word_rate = np.zeros(n_timepoints, dtype=np.float64)
        for timing in word_timings:
            onset_tr = int(float(timing["onset_sec"]) / self.tr)
            if onset_tr < n_timepoints:
                word_rate[onset_tr] += 1.0
        return word_rate

    def ridge_regression_fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        alphas: Optional[List[float]] = None,
    ):
        from sklearn.linear_model import RidgeCV

        alphas = alphas or [1e-2, 1e-1, 1, 10, 100]
        model = RidgeCV(alphas=np.asarray(alphas), cv=5)
        model.fit(X_train, Y_train)
        return model

    def evaluate_alignment(self, model, X_test: np.ndarray, Y_test: np.ndarray) -> Dict:
        from scipy.stats import pearsonr

        Y_pred = model.predict(X_test)
        n_voxels = Y_test.shape[1]
        correlations = []
        for v in range(n_voxels):
            r, _ = pearsonr(Y_pred[:, v], Y_test[:, v])
            correlations.append(float(r))
        correlations_arr = np.asarray(correlations, dtype=np.float64)
        return {
            "mean_r": float(np.mean(correlations_arr)),
            "median_r": float(np.median(correlations_arr)),
            "top10pct_r": float(np.percentile(correlations_arr, 90)),
            "n_voxels_above_0.1": int((correlations_arr > 0.1).sum()),
            "per_voxel_r": correlations_arr.tolist(),
        }

    def train_test_split_temporal(self, X: np.ndarray, Y: np.ndarray, test_fraction: float = 0.2):
        split = int(len(X) * (1 - test_fraction))
        return X[:split], X[split:], Y[:split], Y[split:]

    def word_rate_baseline_predictivity(self, word_rate: np.ndarray, Y: np.ndarray, test_fraction: float = 0.2):
        """
        Baseline encoding using only word-rate (+ drift) features.
        """
        n_t = len(word_rate)
        confounds = self._drift_only(n_t)
        Xb = np.hstack([word_rate.reshape(-1, 1), confounds])
        return self._ridge_metric(Xb, Y, test_fraction)

    def shallow_lexical_baseline(self, word_strings: List[str], Y: np.ndarray, test_fraction: float = 0.2):
        """Bag-of-char-length / word-length proxy baseline (weak lexical surrogate without full BoW)."""
        lengths = np.asarray([len(w) for w in word_strings], dtype=np.float64)
        if lengths.size != len(Y):
            m = min(lengths.size, len(Y))
            lengths = lengths[:m]
            Y = Y[:m]
        n_t = len(lengths)
        confounds = self._drift_only(n_t)
        Xb = np.hstack([lengths.reshape(-1, 1), confounds])
        return self._ridge_metric(Xb, Y, test_fraction)

    def _drift_only(self, n_timepoints: int) -> np.ndarray:
        pos = np.arange(n_timepoints).reshape(-1, 1) / max(n_timepoints, 1)
        return np.hstack([pos, pos**2])

    def _ridge_metric(self, X: np.ndarray, Y: np.ndarray, test_fraction: float) -> Dict:
        X_tr, X_te, Y_tr, Y_te = self.train_test_split_temporal(X, Y, test_fraction)
        if len(X_tr) < 5 or len(X_te) < 2:
            return {"mean_r": float("nan"), "note": "insufficient_timepoints"}
        model = self.ridge_regression_fit(X_tr, Y_tr)
        return self.evaluate_alignment(model, X_te, Y_te)
