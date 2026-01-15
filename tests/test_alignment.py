import numpy as np

from alignment.confound_control import build_confound_matrix, residualize
from alignment.encoding_pipeline import EncodingAlignmentPipeline


def test_align_returns_expected_shape():
    pipe = EncodingAlignmentPipeline(tr=2.0, hrf_lag_trs=2)

    word_timings = [{"onset_sec": 0.0}, {"onset_sec": 2.0}, {"onset_sec": 4.0}]
    hidden_dim = 16
    token_rows = []
    for _ in range(10):
        token_rows.append(
            {
                "activations": {
                    0: np.ones(hidden_dim, dtype=np.float32),
                }
            }
        )

    last_idx = [0, 1, 2]
    n_trs = 6
    feats = pipe.align_token_indices_to_tr(last_idx, token_rows, word_timings, n_trs, layer_idx=0)
    assert feats.shape == (n_trs, hidden_dim)


def test_temporal_split_order():
    pipe = EncodingAlignmentPipeline()
    X = np.arange(10).reshape(-1, 1)
    Y = np.arange(10).reshape(-1, 1)
    Xtr, Xte, Ytr, Yte = pipe.train_test_split_temporal(X, Y, test_fraction=0.2)
    assert Xtr[-1, 0] < Xte[0, 0]


def test_residualize_reduces_variance_proxy():
    rng = np.random.default_rng(0)
    T = 50
    conf = build_confound_matrix(np.ones(T), T)
    # Mostly explained by confounds (+ tiny noise) so residuals shrink
    coef = rng.normal(size=(conf.shape[1], 8))
    X = conf @ coef + rng.normal(scale=0.05, size=(T, 8))
    Xr = residualize(X, conf)
    assert float(np.var(Xr)) < float(np.var(X))
