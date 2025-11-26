#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import joblib

# robust package import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.fdof_pipeline.utils.io import get_logger

logger = get_logger("fdof.analyze_features")

def main():
    model_path = "outputs/artifacts/svm_model.joblib"
    data_path = "data/processed/train.selected.csv"
    label_col = "label"
    top_n = 10

    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading data from {data_path} to get feature names")
    df = pd.read_csv(data_path)
    feature_names = [col for col in df.columns if col != label_col]

    if hasattr(model.named_steps['model'], 'coef_'):
        coefficients = model.named_steps['model'].coef_[0]
    else:
        logger.error("The model does not have coefficients (it might not be a linear model).")
        sys.exit(1)

    feature_importance = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
    feature_importance = feature_importance.sort_values(by='coefficient', ascending=False)

    print("\n--- Top Positive Features ---")
    print(feature_importance.head(top_n))

    print("\n--- Top Negative Features ---")
    print(feature_importance.tail(top_n))

if __name__ == "__main__":
    main()
