from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from ..utils.io import get_logger

logger = get_logger("fdof.outliers")

def threshold_sigma(scores: np.ndarray, e: float) -> np.ndarray:
    """Return boolean mask of outliers where score > mean + e*std."""
    mu = float(scores.mean())
    sd = float(scores.std(ddof=0))
    thr = mu + e * sd
    return scores > thr

def combine_masks(
    md_mask: np.ndarray,
    knn_mask: np.ndarray,
    mode: Literal["intersection", "union"] = "intersection",
) -> np.ndarray:
    if mode == "union":
        return np.logical_or(md_mask, knn_mask)
    return np.logical_and(md_mask, knn_mask)

def apply_outlier_strategy(
    feats_df: pd.DataFrame,
    label_col: str,
    md_scores: np.ndarray,
    knn_scores: np.ndarray,
    md_mask: np.ndarray,
    knn_mask: np.ndarray,
    combine_mode: Literal["intersection", "union"] = "intersection",
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Returns:
      - cleaned_df: dataframe with outliers (combined) removed
      - scores_df: per-row scores + flags
      - combined_mask: boolean array marking rows considered outliers
    """
    combined_mask = combine_masks(md_mask, knn_mask, mode=combine_mode)

    scores_df = pd.DataFrame({
        "md_score": md_scores,
        "knn_score": knn_scores,
        "md_outlier": md_mask.astype(int),
        "knn_outlier": knn_mask.astype(int),
        "combined_outlier": combined_mask.astype(int),
    }, index=feats_df.index)

    cleaned_df = feats_df.loc[~combined_mask].reset_index(drop=True)

    kept = len(cleaned_df)
    dropped = int(combined_mask.sum())
    logger.info(f"Outlier removal -> kept: {kept:,} | removed: {dropped:,}")
    # sanity: keep label intact
    assert label_col in cleaned_df.columns
    return cleaned_df, scores_df, combined_mask

def save_mask_json(path: str | Path, mask: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # store indices of removed rows for reproducibility
    removed_idxs = np.where(mask)[0].tolist()
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"removed_indices": removed_idxs, "total_removed": int(mask.sum())}, f, indent=2)
