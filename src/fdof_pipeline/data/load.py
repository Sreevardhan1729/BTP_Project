from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils.io import get_logger

logger = get_logger("fdof.data")

@dataclass(frozen=True)
class DataConfig:
    train_csv: str
    test_csv: str
    text_col: str
    label_col: str
    drop_cols: Iterable[str]

def _read_csv(path: str) -> pd.DataFrame:
    # Robust read for long, quoted rows
    df = pd.read_csv(
        path,
        encoding="utf-8",
        engine="python",
        on_bad_lines="warn"
    )
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _coerce_binary_label(series: pd.Series) -> pd.Series:
    # Expect {0,1} already; coerce safely
    try:
        ser = pd.to_numeric(series, errors="raise")
    except Exception as e:
        raise ValueError(
            "Label column must be numeric 0/1 in this step. "
            f"Coercion failed with: {e}"
        )
    # Convert floats like 0.0/1.0 to ints
    ser = ser.astype("Int8")
    invalid = ser[~ser.isin([0, 1])]
    if len(invalid) > 0:
        raise ValueError(
            f"Found non-binary labels: {invalid.dropna().unique().tolist()}"
        )
    return ser.astype("int8")

def standardize_frame(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    # Drop nuisance columns if present
    to_drop = [c for c in cfg.drop_cols if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        logger.info(f"Dropped columns: {to_drop}")

    # Check required columns
    for col in [cfg.text_col, cfg.label_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")

    # Keep only the needed columns in order and rename to standard schema
    df = df[[cfg.text_col, cfg.label_col]].rename(
        columns={cfg.text_col: "text", cfg.label_col: "label"}
    )

    # Clean text
    df["text"] = df["text"].map(_clean_text)
    before = len(df)
    df = df[~df["text"].isna() & (df["text"].str.len() > 0)]
    after = len(df)
    if after < before:
        logger.info(f"Removed {before - after:,} empty/NaN text rows")

    # Deduplicate on text
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    if len(df) < before:
        logger.info(f"Removed {before - len(df):,} duplicate texts")

    # Coerce labels to {0,1}
    df["label"] = _coerce_binary_label(df["label"])

    # Final sanity
    assert set(df.columns) == {"text", "label"}
    assert df["label"].isin([0, 1]).all()

    logger.info(f"Standardized frame to shape {df.shape}")
    return df

def load_and_standardize(train_csv: str, test_csv: str, text_col: str, label_col: str, drop_cols: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = DataConfig(train_csv=train_csv, test_csv=test_csv, text_col=text_col, label_col=label_col, drop_cols=drop_cols)
    train_df = _read_csv(cfg.train_csv)
    test_df = _read_csv(cfg.test_csv)

    train_df = standardize_frame(train_df, cfg)
    test_df = standardize_frame(test_df, cfg)

    return train_df, test_df