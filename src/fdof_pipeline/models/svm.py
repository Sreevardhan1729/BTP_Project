from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from joblib import dump
from pathlib import Path
import matplotlib.pyplot as plt

from ..utils.io import get_logger

logger = get_logger("fdof.svm")

@dataclass
class CVConfig:
    n_splits: int = 5
    seed: int = 42

@dataclass
class ModelConfig:
    type: str = "linear_svc"          # "linear_svc" or "svc_rbf"
    class_weight: Optional[str] = None # None or "balanced"
    grid_C: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    grid_gamma: Tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)

def _ensure_xy(df: pd.DataFrame, label_col: str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    feat_cols = [c for c in df.columns if c != label_col]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(df[label_col], errors="raise").to_numpy(dtype=int)
    return X, y, feat_cols

def _build_pipeline(cfg: ModelConfig) -> Pipeline:
    if cfg.type == "svc_rbf":
        svm = SVC(kernel="rbf", probability=False, class_weight=cfg.class_weight, random_state=0)
    else:
        svm = LinearSVC(dual="auto", class_weight=cfg.class_weight, random_state=0, max_iter=10000)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svm", svm),
    ])
    return pipe

def _param_grid(cfg: ModelConfig) -> Dict[str, list]:
    if cfg.type == "svc_rbf":
        return {
            "svm__C": list(cfg.grid_C),
            "svm__gamma": list(cfg.grid_gamma),
        }
    else:
        return {"svm__C": list(cfg.grid_C)}

def cv_search(
    df: pd.DataFrame,
    label_col: str,
    cv_cfg: CVConfig,
    model_cfg: ModelConfig,
) -> Tuple[Pipeline, Dict, float, float]:
    X, y, _ = _ensure_xy(df, label_col)
    pipe = _build_pipeline(model_cfg)
    grid = _param_grid(model_cfg)
    skf = StratifiedKFold(n_splits=cv_cfg.n_splits, shuffle=True, random_state=cv_cfg.seed)

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring="accuracy",
        cv=skf,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X, y)
    y_pred = gs.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    logger.info(f"CV best params: {gs.best_params_} | CV-fit Acc={acc:.4f} F1={f1:.4f}")
    return gs.best_estimator_, gs.best_params_, float(acc), float(f1)

def fit_final(estimator: Pipeline, df: pd.DataFrame, label_col: str) -> Pipeline:
    X, y, _ = _ensure_xy(df, label_col)
    estimator.fit(X, y)
    return estimator

def evaluate(estimator: Pipeline, df: pd.DataFrame, label_col: str) -> Dict[str, float]:
    X, y, _ = _ensure_xy(df, label_col)
    y_pred = estimator.predict(X)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
    }

def save_artifacts(
    estimator: Pipeline,
    test_df: pd.DataFrame,
    label_col: str,
    model_path: str,
    metrics_json: str,
    report_txt: str,
    preds_csv: str,
    cm_png: str,
    cm_csv: str,
) -> None:
    from json import dump as json_dump

    # Ensure dirs
    mp = Path(model_path); mp.parent.mkdir(parents=True, exist_ok=True)
    mj = Path(metrics_json); mj.parent.mkdir(parents=True, exist_ok=True)
    rt = Path(report_txt); rt.parent.mkdir(parents=True, exist_ok=True)
    pc = Path(preds_csv); pc.parent.mkdir(parents=True, exist_ok=True)
    cp = Path(cm_png); cp.parent.mkdir(parents=True, exist_ok=True)
    cc = Path(cm_csv); cc.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    dump(estimator, mp)
    logger.info(f"Saved model -> {mp}")

    # Predictions & metrics
    X, y_true, feat_cols = _ensure_xy(test_df, label_col)
    y_pred = estimator.predict(X)

    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1": float(f1_score(y_true, y_pred)),
        "test_precision": float(precision_score(y_true, y_pred)),
        "test_recall": float(recall_score(y_true, y_pred)),
    }
    with open(mj, "w", encoding="utf-8") as f:
        json_dump(metrics, f, indent=2)

    # Classification report
    rep = classification_report(y_true, y_pred, digits=4)
    with open(rt, "w", encoding="utf-8") as f:
        f.write(rep + "\n")

    # Predictions CSV
    pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    }).to_csv(pc, index=False)

    # Confusion matrix (absolute + normalized)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    cm_norm = confusion_matrix(y_true, y_pred, labels=[0,1], normalize="true")
    # Save CSV
    pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(cc, index=True)

    # Plot PNG (simple Matplotlib, no seaborn)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_title("Normalized Confusion Matrix")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["True 0","True 1"])
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(cp, dpi=160)
    plt.close(fig)

    logger.info(f"Saved metrics -> {mj}")
    logger.info(f"Saved report -> {rt}")
    logger.info(f"Saved preds -> {pc}")
    logger.info(f"Saved confusion matrix -> {cp} | {cc}")
