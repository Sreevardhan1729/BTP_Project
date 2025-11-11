from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

def mahalanobis_scores(
    X: np.ndarray,
    use_ledoit: bool = True,
    ridge: float = 1e-6,
) -> np.ndarray:
    """
    Compute (stable) Mahalanobis distances for each row in X.
    Returns the *distance* (sqrt of quadratic form), shape (n_samples,).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D array.")
    n, d = X.shape
    if n < 2 or d < 1:
        return np.zeros(n, dtype=np.float64)

    # Mean-center
    mu = X.mean(axis=0, dtype=np.float64)
    Xc = X - mu

    if use_ledoit:
        # LedoitWolf is stable for high-dim/collinear features
        lw = LedoitWolf().fit(X)
        cov = lw.covariance_.astype(np.float64, copy=False)
    else:
        # Empirical covariance
        cov = np.cov(X, rowvar=False).astype(np.float64, copy=False)

    # Ridge to ensure invertibility
    cov.flat[:: cov.shape[0] + 1] += ridge

    # Precision via pseudo-inverse for stability
    prec = np.linalg.pinv(cov)

    # MD^2 = (x - mu)^T * prec * (x - mu)
    left = Xc @ prec
    m2 = (left * Xc).sum(axis=1)
    md = np.sqrt(np.maximum(m2, 0.0))
    return md
