#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd

# robust package import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.fdof_pipeline.utils.io import get_logger
from src.fdof_pipeline.models.svm import (
    CVConfig, ModelConfig, cv_search, fit_final, evaluate, save_artifacts
)

logger = get_logger("fdof.train_svm")

def _load(path: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise KeyError(f"Missing label column '{label_col}' in {path}")
    # make sure features are numeric
    feat_cols = [c for c in df.columns if c != label_col]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def main():
    parser = argparse.ArgumentParser(description="Step 6: Train & Evaluate SVM on selected features")
    parser.add_argument("--config", type=str, default="configs/config-svm.yaml",
                        help="Path to SVM config YAML")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inp = cfg["input"]; cv = cfg["cv"]; model = cfg["model"]; trainsec = cfg["train"]; out = cfg["output"]
    label_col = inp.get("label_col", "label")

    train_df = _load(inp["train_csv"], label_col)
    val_df   = _load(inp["val_csv"], label_col)
    test_df  = _load(inp["test_csv"], label_col)

    cv_cfg = CVConfig(n_splits=int(cv.get("n_splits", 5)), seed=int(cv.get("seed", 42)))
    model_cfg = ModelConfig(
        type=str(model.get("type", "linear_svc")),
        class_weight=(None if model.get("class_weight", None) in [None, "null"] else str(model.get("class_weight"))),
        grid_C=tuple(model.get("grid", {}).get("C", [0.25, 0.5, 1.0, 2.0, 4.0])),
        grid_gamma=tuple(model.get("grid", {}).get("gamma", [0.001, 0.01, 0.1, 1.0])),
    )

    # CV search on TRAIN only
    best_estimator, best_params, cv_acc, cv_f1 = cv_search(
        df=train_df, label_col=label_col, cv_cfg=cv_cfg, model_cfg=model_cfg
    )
    logger.info(f"Selected params: {best_params} (CV Acc={cv_acc:.4f}, F1={cv_f1:.4f})")

    # Optionally refit on train+val, then evaluate on test
    if bool(trainsec.get("use_trainval_for_final", True)):
        trainval_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
        final_model = fit_final(best_estimator, trainval_df, label_col)
    else:
        final_model = fit_final(best_estimator, train_df, label_col)

    # Quick validation metrics (sanity)
    val_metrics = evaluate(final_model, val_df, label_col)
    logger.info(f"Val metrics: {val_metrics}")

    # Save artifacts with test evaluation
    save_artifacts(
        estimator=final_model,
        test_df=test_df,
        label_col=label_col,
        model_path=out["model_path"],
        metrics_json=out["metrics_json"],
        report_txt=out["report_txt"],
        preds_csv=out["preds_csv"],
        cm_png=out["cm_png"],
        cm_csv=out["cm_csv"],
    )
    logger.info("SVM training complete.")

if __name__ == "__main__":
    main()
