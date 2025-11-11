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

def _run_svm_scenario(name: str,
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
    model_path = out_dir / f"{name}_svm_model.joblib"
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

def _svm_top_features_from_model(model_path: str, ref_train_csv: str, out_pos_csv: str, out_neg_csv: str,
                                 out_pos_png: str, out_neg_png: str, label_col: str):
    model = joblib.load(model_path)
    # Expect Pipeline(scaler -> svm)
    try:
        svm = model.named_steps["svm"]
    except Exception:
        logger.warning("Model pipeline missing 'svm' step; skipping coefficients plot.")
        return

    if not hasattr(svm, "coef_"):
        logger.info("Non-linear SVM detected (no coef_); skipping top-coefficient plots.")
        return

    # feature names from CSV order
    df = pd.read_csv(ref_train_csv)
    feat_cols = [c for c in df.columns if c != label_col]

    coefs = svm.coef_.ravel()
    # Map to names
    pairs = list(zip(feat_cols, coefs))
    # Top positive/negative
    top_pos = sorted(pairs, key=lambda x: x[1], reverse=True)[:20]
    top_neg = sorted(pairs, key=lambda x: x[1])[:20]

    # Save CSVs
    Path(out_pos_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(top_pos, columns=["feature", "coef"]).to_csv(out_pos_csv, index=False)
    pd.DataFrame(top_neg, columns=["feature", "coef"]).to_csv(out_neg_csv, index=False)

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

    _barh(top_pos, "Top Positive SVM Coefficients (push toward class 1)", out_pos_png)
    _barh(top_neg, "Top Negative SVM Coefficients (push toward class 0)", out_neg_png)

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
    model_cfg = cfg["model"]; cv_cfg = cfg["cv"]

    exp_dir = Path(out["exp_dir"]); exp_dir.mkdir(parents=True, exist_ok=True)

    svm_cv = SVM_CVConfig(n_splits=int(cv_cfg.get("n_splits", 5)),
                          seed=int(cv_cfg.get("seed", 42)))
    svm_model = SVM_ModelConfig(
        type=str(model_cfg.get("type", "linear_svc")),
        class_weight=(None if model_cfg.get("class_weight", None) in [None, "null"] else str(model_cfg.get("class_weight"))),
        grid_C=tuple(model_cfg.get("grid", {}).get("C", [0.25, 0.5, 1.0, 2.0, 4.0])),
        grid_gamma=tuple(model_cfg.get("grid", {}).get("gamma", [0.001, 0.01, 0.1, 1.0])),
    )

    results = []

    # Scenario A: Baseline (no outliers, no selection)
    results.append(_run_svm_scenario(
        "baseline",
        paths["baseline_train"], paths["baseline_val"], paths["baseline_test"],
        label_col, svm_cv, svm_model, exp_dir
    ))

    # Scenario B: Cleaned only (outliers removed)
    results.append(_run_svm_scenario(
        "cleaned_only",
        paths["cleaned_train"], paths["cleaned_val"], paths["cleaned_test"],
        label_col, svm_cv, svm_model, exp_dir
    ))

    # Scenario C: Selected (FDOF full: cleaned + EFSA)
    results.append(_run_svm_scenario(
        "selected",
        paths["selected_train"], paths["selected_val"], paths["selected_test"],
        label_col, svm_cv, svm_model, exp_dir
    ))

    # Ablation: drop engineered features from selected (optional)
    if bool(cfg.get("ablation", {}).get("drop_engineered_from_selected", True)):
        derived_dir = exp_dir / "derived"
        tr, va, te = _derive_selected_wo_engineered(
            paths["selected_train"], paths["selected_val"], paths["selected_test"], label_col, derived_dir
        )
        results.append(_run_svm_scenario(
            "selected_no_engineered", tr, va, te, label_col, svm_cv, svm_model, exp_dir
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
    # SVM top features from Step 6 model (trained on selected)
    if Path(paths["step6_model_path"]).exists():
        _svm_top_features_from_model(
            model_path=paths["step6_model_path"],
            ref_train_csv=paths["selected_train"],
            out_pos_csv=out["coef_top_pos_csv"],
            out_neg_csv=out["coef_top_neg_csv"],
            out_pos_png=out["svm_top_pos_png"],
            out_neg_png=out["svm_top_neg_png"],
            label_col=label_col,
        )

    logger.info(f"Experiment suite complete. Summary -> {out['summary_csv']}")

if __name__ == "__main__":
    main()
