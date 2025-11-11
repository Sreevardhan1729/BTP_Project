#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd

# Make package imports robust when running as a script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.fdof_pipeline.features.build_features import (
    FeaturizerConfig, TfidfCfg, build_and_save_features
)
from src.fdof_pipeline.utils.io import get_logger

logger = get_logger("fdof.featurize")

def _load_df(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(
            f"Expected columns '{text_col}' and '{label_col}' in {path}. Found: {list(df.columns)}"
        )
    return df[[text_col, label_col]].copy()

def main():
    parser = argparse.ArgumentParser(description="Step 2: Build 220-D features")
    parser.add_argument("--config", type=str, default="configs/config-featurize.yaml",
                        help="Path to featurization config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inp = cfg["input"]
    out = cfg["output"]
    fcfg = cfg["featurizer"]

    # Build FeaturizerConfig
    tfidf_cfg = TfidfCfg(
        analyzer=fcfg["tfidf"].get("analyzer", "word"),
        ngram_range=tuple(fcfg["tfidf"].get("ngram_range", [1, 1])),
        lowercase=bool(fcfg["tfidf"].get("lowercase", True)),
        min_df=fcfg["tfidf"].get("min_df", 2),
        max_df=fcfg["tfidf"].get("max_df", 1.0),
        max_features=int(fcfg["tfidf"].get("max_features_hint", 200)),
    )
    feat_cfg = FeaturizerConfig(
        target_dim=int(fcfg.get("target_dim", 220)),
        engineered_dim=int(fcfg.get("engineered_dim", 20)),
        tfidf=tfidf_cfg,
    )

    # Load data
    text_col = inp.get("text_col", "text")
    label_col = inp.get("label_col", "label")
    train_df = _load_df(inp["train_csv"], text_col, label_col)
    val_df   = _load_df(inp["val_csv"], text_col, label_col)
    test_df  = _load_df(inp["test_csv"], text_col, label_col)

    logger.info(f"Train/Val/Test shapes: {train_df.shape} | {val_df.shape} | {test_df.shape}")

    # Execute build + save
    build_and_save_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cfg=feat_cfg,
        out_dir=out["features_dir"],
        out_train=out["train_out"],
        out_val=out["val_out"],
        out_test=out["test_out"],
        names_json=out["feature_names_json"],
        vec_path=out["vectorizer_path"],
        text_col=text_col,
        label_col=label_col,
    )

    logger.info("Featurization complete.")

if __name__ == "__main__":
    main()
