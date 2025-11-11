from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score

from ..utils.io import get_logger
from ..outliers.mahalanobis import mahalanobis_scores
from ..outliers.knn import knn_outlier_scores
from ..outliers.integrate import threshold_sigma, combine_masks

logger = get_logger("fdof.cv")

@dataclass
class CVConfig:
    n_splits: int = 5
    seed: int = 42
    standardize: bool = True

@dataclass
class SearchSpace:
    md_e_range: Tuple[float, float] = (1.5, 4.0)
    knn_e_range: Tuple[float, float] = (1.0, 4.0)
    k_range: Tuple[int, int] = (5, 50)
    combine_choices: Tuple[str, ...] = ("intersection", "union")
    knn_score_choices: Tuple[str, ...] = ("avg_k_distance", "kth_distance")

@dataclass
class ClassifierCfg:
    type: str = "linear_svc"  # "linear_svc" or "svc_rbf"
    C: float = 1.0

def _ensure_numeric(df: pd.DataFrame, label_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feat_cols = [c for c in df.columns if c != label_col]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    y = pd.to_numeric(df[label_col], errors="raise").to_numpy(dtype=np.int32)
    return X, y, feat_cols

def _train_eval_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    params: Dict,
    clf_cfg: ClassifierCfg,
    standardize: bool,
) -> Tuple[float, float]:
    """
    Apply outlier removal on training split, fit classifier, evaluate on validation split.
    """
    # Standardize features for outlier scoring (and later classifier)
    scaler = None
    X_tr_proc = X_tr
    X_va_proc = X_va
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr_proc = scaler.fit_transform(X_tr)
        X_va_proc = scaler.transform(X_va)

    # --- Outlier detection on TRAIN only ---
    k = int(params["k"])
    k = max(1, min(k, X_tr_proc.shape[0] - 1)) if X_tr_proc.shape[0] > 1 else 1

    md_scores = mahalanobis_scores(
        X=X_tr_proc,
        use_ledoit=True,
        ridge=1e-6,
    )
    md_mask = threshold_sigma(md_scores, e=float(params["e_md"]))

    knn_scores = knn_outlier_scores(
        X=X_tr_proc,
        k=k,
        metric="minkowski",
        p=2,
        mode=str(params["knn_score"]),
    )
    knn_mask = threshold_sigma(knn_scores, e=float(params["e_knn"]))

    combined_mask = combine_masks(md_mask, knn_mask, mode=str(params["combine"]))
    # Keep non-outliers
    keep_idx = np.where(~combined_mask)[0]
    if keep_idx.size == 0:
        # If everything got dropped (rare), fall back to no removal for this fold to avoid errors
        X_tr_clean, y_tr_clean = X_tr_proc, y_tr
    else:
        X_tr_clean, y_tr_clean = X_tr_proc[keep_idx], y_tr[keep_idx]

    # --- Classifier ---
    if clf_cfg.type == "svc_rbf":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0)
    else:
        clf = LinearSVC(C=float(clf_cfg.C), random_state=0, max_iter=10000)

    clf.fit(X_tr_clean, y_tr_clean)
    y_pred = clf.predict(X_va_proc)

    acc = accuracy_score(y_va, y_pred)
    f1 = f1_score(y_va, y_pred, average="binary")
    return acc, f1

def crossval_score_params(
    df: pd.DataFrame,
    label_col: str,
    params: Dict,
    cv_cfg: CVConfig,
    clf_cfg: ClassifierCfg,
) -> Tuple[float, float]:
    """
    Returns (mean_acc, mean_f1) across CV folds for the given outlier params.
    """
    X, y, _ = _ensure_numeric(df, label_col)

    skf = StratifiedKFold(n_splits=cv_cfg.n_splits, shuffle=True, random_state=cv_cfg.seed)
    accs: List[float] = []
    f1s: List[float] = []

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        acc, f1 = _train_eval_fold(
            X_tr=X_tr,
            y_tr=y_tr,
            X_va=X_va,
            y_va=y_va,
            params=params,
            clf_cfg=clf_cfg,
            standardize=cv_cfg.standardize,
        )
        accs.append(acc)
        f1s.append(f1)

    mean_acc = float(np.mean(accs)) if accs else 0.0
    mean_f1 = float(np.mean(f1s)) if f1s else 0.0
    return mean_acc, mean_f1

def apply_best_params_full(
    df: pd.DataFrame,
    label_col: str,
    best_params: Dict,
    standardize: bool,
) -> pd.DataFrame:
    """
    Apply best outlier params once to the FULL training set and return the cleaned DataFrame.
    """
    X, y, feat_cols = _ensure_numeric(df, label_col)

    scaler = None
    X_proc = X
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_proc = scaler.fit_transform(X)

    # K may not exceed n-1
    k = int(best_params["k"])
    k = max(1, min(k, X_proc.shape[0] - 1)) if X_proc.shape[0] > 1 else 1

    md_scores = mahalanobis_scores(X_proc, use_ledoit=True, ridge=1e-6)
    md_mask = threshold_sigma(md_scores, e=float(best_params["e_md"]))

    knn_scores = knn_outlier_scores(
        X=X_proc,
        k=k,
        metric="minkowski",
        p=2,
        mode=str(best_params["knn_score"]),
    )
    knn_mask = threshold_sigma(knn_scores, e=float(best_params["e_knn"]))

    combined_mask = combine_masks(md_mask, knn_mask, mode=str(best_params["combine"]))
    keep_idx = np.where(~combined_mask)[0]
    if keep_idx.size == 0:
        logger.warning("Best params removed all rows; returning original df.")
        return df.copy()

    cleaned_df = df.iloc[keep_idx].reset_index(drop=True)
    return cleaned_df

def save_tuning_report(path: str, best_params: Dict, best_scores: Dict, trials: List[Dict]) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params": best_params,
                "best_scores": best_scores,
                "trials": trials,
            },
            f,
            indent=2,
        )
