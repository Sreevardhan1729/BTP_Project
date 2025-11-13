#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json
import yaml
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# robust package import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.fdof_pipeline.utils.io import get_logger
from src.fdof_pipeline.models.svm import (
    CVConfig as SVM_CVConfig,
    ModelConfig as SVM_ModelConfig,
    cv_search, fit_final, evaluate
)
from src.fdof_pipeline.features.build_features import ENGINEERED_ORDER

logger = get_logger("fdof.experiments")

def _load_df(path: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise KeyError(f"Missing '{label_col}' in {path}")
    feat_cols = [c for c in df.columns if c != label_col]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def _run_scenario(name: str,
                      train_csv: str, val_csv: str, test_csv: str, label_col: str,
                      cv_cfg: SVM_CVConfig, model_cfg: SVM_ModelConfig,
                      out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = _load_df(train_csv, label_col)
    val_df   = _load_df(val_csv, label_col)
    test_df  = _load_df(test_csv, label_col)

    best_est, best_params, cv_acc, cv_f1 = cv_search(
        df=train_df, label_col=label_col, cv_cfg=cv_cfg, model_cfg=model_cfg
    )
    logger.info(f"[{name}] best params: {best_params} (CV Acc={cv_acc:.4f}, F1={cv_f1:.4f})")

    final_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    final_model = fit_final(best_est, final_df, label_col)
    test_metrics = evaluate(final_model, test_df, label_col)

    # Save artifacts, but scenario-specific
    model_path = out_dir / f"{name}_model.joblib"
    joblib.dump(final_model, model_path)

    preds = final_model.predict(test_df[[c for c in test_df.columns if c != label_col]].to_numpy())
    pd.DataFrame({"y_true": test_df[label_col].to_numpy(), "y_pred": preds}).to_csv(out_dir / f"{name}_preds.csv", index=False)

    with open(out_dir / f"{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "cv_best_params": best_params,
            "cv_acc": float(cv_acc),
            "cv_f1": float(cv_f1),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_f1": float(test_metrics["f1"]),
            "test_precision": float(test_metrics["precision"]),
            "test_recall": float(test_metrics["recall"]),
        }, f, indent=2)

    return {
        "scenario": name,
        "model": model_cfg.type,
        "cv_acc": float(cv_acc),
        "cv_f1": float(cv_f1),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_f1": float(test_metrics["f1"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "model_path": str(model_path),
    }

def _derive_selected_wo_engineered(train_csv: str, val_csv: str, test_csv: str, label_col: str,
                                   out_base_dir: Path) -> tuple[str, str, str]:
    """Create datasets by dropping engineered features from the selected sets."""
    out_base_dir.mkdir(parents=True, exist_ok=True)
    eng_set = set(ENGINEERED_ORDER)

    def _drop(path: str, tag: str) -> str:
        df = _load_df(path, label_col)
        feat_cols = [c for c in df.columns if c != label_col]
        keep_cols = [c for c in feat_cols if c not in eng_set]
        out_path = out_base_dir / f"{tag}.csv"
        pd.concat([df[keep_cols], df[[label_col]]], axis=1).to_csv(out_path, index=False)
        return str(out_path)

    tr = _drop(train_csv, "train.selected.noeng")
    va = _drop(val_csv, "val.selected.noeng")
    te = _drop(test_csv, "test.selected.noeng")
    return tr, va, te

def _plot_efsa_progress(hist_csv: str, out_png: str):
    hist = pd.read_csv(hist_csv)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(hist["generation"], hist["best_score"], label="best")
    ax.plot(hist["generation"], hist["mean_score"], label="mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("CV score")
    ax.set_title("EFSA Progress")
    ax.legend()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def _plot_outlier_trials(trials_csv: str, out_k_vs_acc: str, out_box: str):
    df = pd.read_csv(trials_csv)
    # Scatter: k vs mean_acc (color by combine)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    for mode, g in df.groupby(df["params"].apply(lambda x: json.loads(x.replace("'", '"'))["combine"] if isinstance(x, str) else "")):
        # parse params JSON-ish string
        if isinstance(mode, str):
            ks = g["params"].apply(lambda x: json.loads(x.replace("'", '"'))["k"] if isinstance(x, str) else np.nan)
            ax1.scatter(ks, g["mean_acc"], label=mode, alpha=0.7)
    ax1.set_xlabel("k (neighbors)")
    ax1.set_ylabel("CV mean accuracy")
    ax1.set_title("Outlier Trials: k vs Acc (colored by combine)")
    ax1.legend()
    Path(out_k_vs_acc).parent.mkdir(parents=True, exist_ok=True)
    fig1.tight_layout()
    fig1.savefig(out_k_vs_acc, dpi=160)
    plt.close(fig1)

    # Box by combine
    # Extract combine to a column
    def _combine(row):
        p = row.get("params")
        if isinstance(p, str):
            return json.loads(p.replace("'", '"'))["combine"]
        return ""
    df["combine"] = df.apply(_combine, axis=1)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    groups = [g["mean_acc"].values for _, g in df.groupby("combine")]
    labels = [m for m, _ in df.groupby("combine")]
    ax2.boxplot(groups, labels=labels)
    ax2.set_ylabel("CV mean accuracy")
    ax2.set_title("Outlier Trials: Accuracy by Combine Mode")
    fig2.tight_layout()
    fig2.savefig(out_box, dpi=160)
    plt.close(fig2)

def _top_features_from_model(model_path: str, ref_train_csv: str, out_pos_csv: str, out_neg_csv: str,
                                 out_pos_png: str, out_neg_png: str, label_col: str):
    model = joblib.load(model_path)
    # Expect Pipeline(scaler -> model)
    try:
        inner_model = model.named_steps["model"]
    except Exception:
        logger.warning("Model pipeline missing 'model' step; skipping coefficients plot.")
        return

    if hasattr(inner_model, "coef_"):
        # Linear models
        coefs = inner_model.coef_.ravel()
    elif hasattr(inner_model, "feature_importances_"):
        # Tree-based models
        coefs = inner_model.feature_importances_
    else:
        logger.info("Model does not have 'coef_' or 'feature_importances_'; skipping top-coefficient plots.")
        return

    # feature names from CSV order
    df = pd.read_csv(ref_train_csv)
    feat_cols = [c for c in df.columns if c != label_col]

    # Map to names
    pairs = list(zip(feat_cols, coefs))
    # Top positive/negative
    top_pos = sorted(pairs, key=lambda x: x[1], reverse=True)[:20]
    top_neg = sorted(pairs, key=lambda x: x[1])[:20]

    # Save CSVs
    Path(out_pos_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(top_pos, columns=["feature", "importance"]).to_csv(out_pos_csv, index=False)
    pd.DataFrame(top_neg, columns=["feature", "importance"]).to_csv(out_neg_csv, index=False)

    # Plots
    def _barh(data, title, out_png):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        names = [n for n, _ in data][::-1]
        vals = [v for _, v in data][::-1]
        ax.barh(range(len(vals)), vals)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)

    _barh(top_pos, "Top 20 Positive Features", out_pos_png)
    _barh(top_neg, "Top 20 Negative Features", out_neg_png)

def main():
    parser = argparse.ArgumentParser(description="Step 8: Experiment suite, ablations, explainability")
    parser.add_argument("--config", type=str, default="configs/config-experiments.yaml",
                        help="Path to experiments config YAML")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    label_col = cfg.get("label_col", "label")
    out = cfg["output"]
    model_cfg_data = cfg["model"]; cv_cfg_data = cfg["cv"]

    exp_dir = Path(out["exp_dir"]); exp_dir.mkdir(parents=True, exist_ok=True)

    cv_cfg = SVM_CVConfig(n_splits=int(cv_cfg_data.get("n_splits", 5)),
                          seed=int(cv_cfg_data.get("seed", 42)))
    
    results = []
    scenarios = {
        "baseline": (paths["baseline_train"], paths["baseline_val"], paths["baseline_test"]),
        "cleaned_only": (paths["cleaned_train"], paths["cleaned_val"], paths["cleaned_test"]),
        "selected": (paths["selected_train"], paths["selected_val"], paths["selected_test"]),
    }
    if bool(cfg.get("ablation", {}).get("drop_engineered_from_selected", True)):
        scenarios["selected_no_engineered"] = _derive_selected_wo_engineered(
            paths["selected_train"], paths["selected_val"], paths["selected_test"], label_col, exp_dir / "derived"
        )

    for model_type in ["linear_svc", "random_forest", "xgboost"]:
        
        grid_data = model_cfg_data.get("grid", {})
        if model_type == "random_forest":
            grid_data = cfg.get("rf", {}).get("grid", {})
        elif model_type == "xgboost":
            grid_data = cfg.get("xgboost", {}).get("grid", {})

        current_model_cfg = SVM_ModelConfig(
            type=model_type,
            class_weight=(None if model_cfg_data.get("class_weight", None) in [None, "null"] else str(model_cfg_data.get("class_weight"))),
            grid_C=tuple(grid_data.get("C", [0.25, 0.5, 1.0, 2.0, 4.0])),
            grid_gamma=tuple(grid_data.get("gamma", [0.001, 0.01, 0.1, 1.0])),
            grid_n_estimators=tuple(grid_data.get("n_estimators", [100, 200, 300])),
            grid_max_depth=tuple(grid_data.get("max_depth", [10, 20, None])),
            grid_min_samples_leaf=tuple(grid_data.get("min_samples_leaf", [1, 2, 4])),
            grid_learning_rate=tuple(grid_data.get("learning_rate", [0.05, 0.1, 0.2])),
            grid_subsample=tuple(grid_data.get("subsample", [0.8, 1.0])),
        )

        for name, (train_csv, val_csv, test_csv) in scenarios.items():
            scenario_name = f"{name}_{model_type}"
            results.append(_run_scenario(
                scenario_name,
                train_csv, val_csv, test_csv,
                label_col, cv_cfg, current_model_cfg, exp_dir
            ))

    # Write summary table
    summary = pd.DataFrame(results)
    Path(out["summary_csv"]).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out["summary_csv"], index=False)

    # Explainability plots
    if Path(paths["efsa_history_csv"]).exists():
        _plot_efsa_progress(paths["efsa_history_csv"], out["efsa_progress_png"])
    if Path(paths["outlier_trials_csv"]).exists():
        _plot_outlier_trials(paths["outlier_trials_csv"],
                             out["outlier_trials_k_vs_acc_png"],
                             out["outlier_trials_box_acc_png"])
    # Top features from selected models
    for model_type in ["linear_svc", "random_forest", "xgboost"]:
        model_path = exp_dir / f"selected_{model_type}_model.joblib"
        if model_path.exists():
            _top_features_from_model(
                model_path=str(model_path),
                ref_train_csv=paths["selected_train"],
                out_pos_csv=out["coef_top_pos_csv"].replace(".csv", f"_{model_type}.csv"),
                out_neg_csv=out["coef_top_neg_csv"].replace(".csv", f"_{model_type}.csv"),
                out_pos_png=out["top_pos_png"].replace(".png", f"_{model_type}.png"),
                out_neg_png=out["top_neg_png"].replace(".png", f"_{model_type}.png"),
                label_col=label_col,
            )

    logger.info(f"Experiment suite complete. Summary -> {out['summary_csv']}")

if __name__ == "__main__":
    main()