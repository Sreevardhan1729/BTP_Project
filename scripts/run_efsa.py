#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# robust package import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.fdof_pipeline.utils.io import get_logger
from src.fdof_pipeline.selection.chromosome import GAParams
from src.fdof_pipeline.selection.fitness import CVCfg, ClassifierCfg
from src.fdof_pipeline.selection.efsa import (
    EFSAConfig, EFSA, filter_and_save_by_mask
)

logger = get_logger("fdof.run_efsa")

def _load_df(path: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' missing in {path}.")
    # numeric conversion for all features (keep label as is)
    feat_cols = [c for c in df.columns if c != label_col]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def main():
    parser = argparse.ArgumentParser(description="Step 5: EFSA Genetic Algorithm for Feature Selection")
    parser.add_argument("--config", type=str, default="configs/config-efsa.yaml",
                        help="Path to EFSA config YAML")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inp = cfg["input"]
    cv_cfg = cfg["cv"]
    ga_cfg = cfg["ga"]
    clf_cfg = cfg["classifier"]
    out = cfg["output"]

    label_col = inp.get("label_col", "label")
    train_df = _load_df(inp["train_csv"], label_col)
    val_df   = _load_df(inp["val_csv"], label_col)
    test_df  = _load_df(inp["test_csv"], label_col)

    # Ensure train/val/test share the same feature set
    feat_cols = [c for c in train_df.columns if c != label_col]
    for df, name in [(val_df, "val"), (test_df, "test")]:
        cols = [c for c in df.columns if c != label_col]
        if cols != feat_cols:
            # align column order and missing columns if any
            missing = [c for c in feat_cols if c not in df.columns]
            extras = [c for c in cols if c not in feat_cols]
            if missing:
                raise ValueError(f"{name} set missing columns not in train: {missing[:5]}...")
            if extras:
                # drop any accidental extras
                df.drop(columns=extras, inplace=True)
            df = df[feat_cols + [label_col]]
            if name == "val": val_df = df
            else: test_df = df

    # Build configs
    efsa_conf = EFSAConfig(
        ga=GAParams(
            pop_size=int(ga_cfg.get("pop_size", 40)),
            generations=int(ga_cfg.get("generations", 30)),
            tournament_size=int(ga_cfg.get("tournament_size", 3)),
            crossover_prob=float(ga_cfg.get("crossover_prob", 0.9)),
            mutation_prob=float(ga_cfg.get("mutation_prob", 0.01)),
            min_features=int(ga_cfg.get("min_features", 10)),
            early_stop_patience=int(ga_cfg.get("early_stop_patience", 6)),
        ),
        cv=CVCfg(
            n_splits=int(cv_cfg.get("n_splits", 5)),
            seed=int(cv_cfg.get("seed", 42)),
            standardize=bool(cv_cfg.get("standardize", True)),
        ),
        clf=ClassifierCfg(
            type=str(clf_cfg.get("type", "linear_svc")),
            C=float(clf_cfg.get("C", 1.0)),
        ),
        maximize_metric=str(ga_cfg.get("maximize_metric", "accuracy")),
        seed=int(cv_cfg.get("seed", 42)),
    )

    # Run EFSA
    efsa = EFSA(df=train_df, label_col=label_col, cfg=efsa_conf)
    best_mask, best_score = efsa.run()
    logger.info(f"EFSA best {efsa_conf.maximize_metric}={best_score:.4f} with {int(best_mask.sum())}/{efsa.n_features} features.")

    # Save artifacts
    efsa.save_outputs(
        best_mask=best_mask,
        out_selected_json=out["selected_features_json"],
        out_mask_npy=out["mask_npy"],
        out_history_csv=out["ga_history_csv"],
        out_report_json=out["ga_report_json"],
    )

    # Filter and save CSVs
    filter_and_save_by_mask(
        mask=best_mask,
        label_col=label_col,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        out_train=out["selected_train_csv"],
        out_val=out["selected_val_csv"],
        out_test=out["selected_test_csv"],
    )

    logger.info("EFSA feature selection complete.")

if __name__ == "__main__":
    main()
