#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Make package imports robust when running as a script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.fdof_pipeline.utils.io import get_logger
from src.fdof_pipeline.outliers.mahalanobis import mahalanobis_scores
from src.fdof_pipeline.outliers.knn import knn_outlier_scores
from src.fdof_pipeline.outliers.integrate import (
    threshold_sigma,
    apply_outlier_strategy,
    save_mask_json,
)

logger = get_logger("fdof.apply_outliers")

def _load_features(path: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in {path} (cols={list(df.columns)[:8]}...)")
    # ensure numeric features
    feat_cols = [c for c in df.columns if c != label_col]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def main():
    parser = argparse.ArgumentParser(description="Step 3: Integrated Outlier Detection (MD + KNN)")
    parser.add_argument("--config", type=str, default="configs/config-outliers.yaml",
                        help="Path to outlier config YAML")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inp = cfg["input"]
    outliers = cfg["outliers"]
    out = cfg["output"]

    label_col = inp.get("label_col", "label")
    train_path = inp["features_train_csv"]
    feats_df = _load_features(train_path, label_col)

    feat_cols = [c for c in feats_df.columns if c != label_col]
    X = feats_df[feat_cols].to_numpy(dtype=np.float64, copy=True)

    # Standardize if requested
    if bool(outliers.get("standardize", True)):
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X)

    # MD scores + mask
    md_cfg = outliers.get("md", {})
    md_scores = mahalanobis_scores(
        X=X,
        use_ledoit=bool(md_cfg.get("use_ledoit", True)),
        ridge=float(md_cfg.get("ridge", 1e-6)),
    )
    md_mask = threshold_sigma(md_scores, e=float(md_cfg.get("threshold_sigma_e", 2.5)))

    # KNN scores + mask
    knn_cfg = outliers.get("knn", {})
    k = int(knn_cfg.get("k", 10))
    # Clamp k to at most n-1
    k = max(1, min(k, X.shape[0] - 1)) if X.shape[0] > 1 else 1

    knn_scores = knn_outlier_scores(
        X=X,
        k=k,
        metric=str(knn_cfg.get("metric", "minkowski")),
        p=int(knn_cfg.get("p", 2)),
        mode=str(knn_cfg.get("score", "avg_k_distance")),
    )
    knn_mask = threshold_sigma(knn_scores, e=float(knn_cfg.get("threshold_sigma_e", 2.0)))

    # Combine & produce cleaned train
    mode = str(outliers.get("combine", "intersection")).lower()
    cleaned_df, scores_df, combined_mask = apply_outlier_strategy(
        feats_df=feats_df,
        label_col=label_col,
        md_scores=md_scores,
        knn_scores=knn_scores,
        md_mask=md_mask,
        knn_mask=knn_mask,
        combine_mode=mode,
    )

    # Save outputs
    cleaned_out = Path(out["cleaned_train_out"])
    cleaned_out.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(cleaned_out, index=False)

    scores_out = Path(out["scores_csv"])
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    # include original row index for traceability
    scores_df = scores_df.reset_index().rename(columns={"index": "row_index"})
    scores_df.to_csv(scores_out, index=False)

    save_mask_json(out["mask_json"], combined_mask)

    logger.info(f"Saved cleaned train: {cleaned_out}")
    logger.info(f"Saved scores: {scores_out}")
    logger.info(f"Saved mask JSON: {out['mask_json']}")
    logger.info("Outlier application complete.")

if __name__ == "__main__":
    main()
