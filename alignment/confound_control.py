import numpy as np
from sklearn.linear_model import Ridge


def residualize(X: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    model = Ridge(alpha=1.0)
    model.fit(confounds, X)
    X_pred = model.predict(confounds)
    return X - X_pred


def build_confound_matrix(
    word_rate: np.ndarray,
    n_timepoints: int,
    include_position: bool = True,
) -> np.ndarray:
    confounds = [word_rate.reshape(-1, 1)]
    if include_position:
        pos = np.arange(n_timepoints).reshape(-1, 1) / max(n_timepoints, 1)
        confounds.append(pos)
        confounds.append(pos**2)
    return np.hstack(confounds)
