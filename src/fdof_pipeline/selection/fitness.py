from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal, Dict
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score

@dataclass
class CVCfg:
    n_splits: int = 5
    seed: int = 42
    standardize: bool = True

@dataclass
class ClassifierCfg:
    type: str = "linear_svc"  # or "svc_rbf"
    C: float = 1.0

def ensure_xy(df: pd.DataFrame, label_col: str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    feat_cols = [c for c in df.columns if c != label_col]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(df[label_col], errors="raise").to_numpy(dtype=int)
    return X, y, feat_cols

def cv_metric_for_mask(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    cv: CVCfg,
    clf_cfg: ClassifierCfg,
    metric: Literal["accuracy", "f1"] = "accuracy",
) -> float:
    """Compute mean CV metric for a feature subset mask."""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.sum() == 0:
        return 0.0

    Xm = X[:, mask]
    skf = StratifiedKFold(n_splits=cv.n_splits, shuffle=True, random_state=cv.seed)
    scores = []
    for tr, va in skf.split(Xm, y):
        Xtr, Xva = Xm[tr], Xm[va]
        ytr, yva = y[tr], y[va]

        if cv.standardize:
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xtr = scaler.fit_transform(Xtr)
            Xva = scaler.transform(Xva)

        if clf_cfg.type == "svc_rbf":
            clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0)
        else:
            clf = LinearSVC(C=float(clf_cfg.C), random_state=0, max_iter=10000)

        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xva)

        if metric == "f1":
            scores.append(f1_score(yva, yhat, average="binary"))
        else:
            scores.append(accuracy_score(yva, yhat))
    return float(np.mean(scores)) if scores else 0.0
