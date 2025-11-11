#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
import json
import pandas as pd
import numpy as np
import optuna

# make package imports robust when running as a script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.fdof_pipeline.utils.io import get_logger
from src.fdof_pipeline.training.cv_pipeline import (
    CVConfig, SearchSpace, ClassifierCfg,
    crossval_score_params, apply_best_params_full, save_tuning_report
)

logger = get_logger("fdof.tune")

def _load_train(path: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in {path}.")
    return df.copy()

def main():
    parser = argparse.ArgumentParser(description="Step 4: 5-fold CV tuning for outlier params")
    parser.add_argument("--config", type=str, default="configs/config-tune-outliers.yaml",
                        help="Path to tuning config YAML")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inp = cfg["input"]
    cv_cfg = cfg["cv"]
    search = cfg["search"]
    clf_cfg = cfg["classifier"]
    out = cfg["output"]

    label_col = inp.get("label_col", "label")
    df = _load_train(inp["features_train_csv"], label_col)

    cvconf = CVConfig(
        n_splits=int(cv_cfg.get("n_splits", 5)),
        seed=int(cv_cfg.get("seed", 42)),
        standardize=bool(cv_cfg.get("standardize", True)),
    )
    space = SearchSpace(
        md_e_range=tuple(search.get("md_e_range", [1.5, 4.0])),
        knn_e_range=tuple(search.get("knn_e_range", [1.0, 4.0])),
        k_range=tuple(search.get("k_range", [5, 50])),
        combine_choices=tuple(search.get("combine_choices", ["intersection", "union"])),
        knn_score_choices=tuple(search.get("knn_score_choices", ["avg_k_distance", "kth_distance"])),
    )
    clfconf = ClassifierCfg(
        type=str(clf_cfg.get("type", "linear_svc")),
        C=float(clf_cfg.get("C", 1.0)),
    )

    n_trials = int(search.get("n_trials", 30))
    trials_log = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "e_md": trial.suggest_float("e_md", space.md_e_range[0], space.md_e_range[1], step=0.1),
            "e_knn": trial.suggest_float("e_knn", space.knn_e_range[0], space.knn_e_range[1], step=0.1),
            "k": trial.suggest_int("k", space.k_range[0], space.k_range[1]),
            "combine": trial.suggest_categorical("combine", list(space.combine_choices)),
            "knn_score": trial.suggest_categorical("knn_score", list(space.knn_score_choices)),
        }
        mean_acc, mean_f1 = crossval_score_params(
            df=df, label_col=label_col, params=params, cv_cfg=cvconf, clf_cfg=clfconf
        )
        # record the trial
        trials_log.append({
            "trial_number": trial.number,
            "params": params,
            "mean_acc": mean_acc,
            "mean_f1": mean_f1,
        })
        # maximize accuracy
        return mean_acc

    study = optuna.create_study(direction="maximize", study_name="outlier_param_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_value = float(study.best_value)
    # recompute F1 for best (since objective returns only accuracy)
    mean_acc, mean_f1 = crossval_score_params(
        df=df, label_col=label_col, params=best_params, cv_cfg=cvconf, clf_cfg=clfconf
    )
    logger.info(f"Best params: {best_params} | CV Acc={mean_acc:.4f} | CV F1={mean_f1:.4f}")

    # Apply best params on FULL training set and save cleaned CSV
    cleaned_df = apply_best_params_full(
        df=df, label_col=label_col, best_params=best_params, standardize=cvconf.standardize
    )
    cleaned_out = Path(out["cleaned_train_out"])
    cleaned_out.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(cleaned_out, index=False)

    # Save tuning report JSON
    report_json = out["tuning_report_json"]
    save_tuning_report(
        path=report_json,
        best_params=best_params,
        best_scores={"cv_mean_acc": mean_acc, "cv_mean_f1": mean_f1},
        trials=trials_log,
    )

    # Save per-trial CSV
    trials_csv = Path(out["trial_logs_csv"])
    trials_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials_log).to_csv(trials_csv, index=False)

    logger.info(f"Saved cleaned train: {cleaned_out}")
    logger.info(f"Saved tuning report: {report_json}")
    logger.info(f"Saved trial logs: {trials_csv}")
    logger.info("Outlier tuning complete.")

if __name__ == "__main__":
    main()
