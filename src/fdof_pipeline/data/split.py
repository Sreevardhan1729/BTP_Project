from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.io import get_logger

logger = get_logger("fdof.split")

def stratified_train_val(
    df: pd.DataFrame, val_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not {"text", "label"}.issubset(df.columns):
        raise KeyError("Expected columns: {'text','label'}")

    tr_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["label"],
        shuffle=True,
    )
    logger.info(
        f"Stratified split -> train: {len(tr_df):,}, val: {len(val_df):,}"
    )
    return tr_df.reset_index(drop=True), val_df.reset_index(drop=True)
