from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

def knn_outlier_scores(
    X: np.ndarray,
    k: int = 10,
    metric: str = "minkowski",
    p: int = 2,
    mode: str = "avg_k_distance",
) -> np.ndarray:
    """
    Compute a KNN-based outlier score per sample.
    - mode="avg_k_distance": mean distance to the k nearest neighbors (excl. self)
    - mode="kth_distance": distance to the k-th neighbor (excl. self)
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D array.")
    n = X.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    k_eff = max(1, min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, p=p if metric == "minkowski" else None)
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)

    # drop self-distance (0)
    neigh_dists = dists[:, 1:]  # shape: (n, k_eff)

    if mode == "kth_distance":
        # distance to the farthest among the k neighbors
        scores = neigh_dists[:, -1]
    else:
        # average distance to k neighbors
        scores = neigh_dists.mean(axis=1)

    return scores.astype(np.float64, copy=False)
