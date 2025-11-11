from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from joblib import dump

from ..utils.io import get_logger, ensure_dir
from .style import style_features
from .complexity import complexity_features
from .psychology import psychology_features

logger = get_logger("fdof.features")

ENGINEERED_ORDER = [
    # style (16)
    "char_len",
    "word_count",
    "avg_word_len",
    "unique_word_ratio",
    "stopword_ratio",
    "punct_ratio",
    "uppercase_char_ratio",
    "uppercase_word_ratio",
    "digit_ratio",
    "url_count",
    "exclamation_count",
    "question_count",
    "comma_count",
    "period_count",
    "colon_semi_count",
    "quote_count",
    # complexity (3)
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog_index",
    # psychology (1)
    "vader_compound",
]
# Ensure exactly 20 engineered
assert len(ENGINEERED_ORDER) == 20, "Engineered feature list must have 20 entries."

@dataclass
class TfidfCfg:
    analyzer: str = "word"
    ngram_range: Tuple[int, int] = (1, 1)
    lowercase: bool = True
    min_df: int | float = 2
    max_df: int | float = 1.0
    max_features: int = 200  # will be set by (target - engineered)

@dataclass
class FeaturizerConfig:
    target_dim: int = 220
    engineered_dim: int = 20
    tfidf: TfidfCfg = field(default_factory=TfidfCfg)

def _engineered_df(texts: List[str]) -> pd.DataFrame:
    rows = []
    for t in texts:
        f = {}
        f.update(style_features(t))
        f.update(complexity_features(t))
        f.update(psychology_features(t))
        # Select and order
        rows.append({k: f.get(k, 0.0) for k in ENGINEERED_ORDER})
    return pd.DataFrame(rows, columns=ENGINEERED_ORDER)

class FeatureBuilder:
    def __init__(self, cfg: FeaturizerConfig):
        self.cfg = cfg
        # Determine TF-IDF size
        self.tfidf_dim = max(self.cfg.target_dim - self.cfg.engineered_dim, 0)
        self.vectorizer: TfidfVectorizer | None = None

    def _make_vectorizer(self, override: Dict | None = None) -> TfidfVectorizer:
        params = {
            "analyzer": self.cfg.tfidf.analyzer,
            "ngram_range": self.cfg.tfidf.ngram_range,
            "lowercase": self.cfg.tfidf.lowercase,
            "min_df": self.cfg.tfidf.min_df,
            "max_df": self.cfg.tfidf.max_df,
            "max_features": self.tfidf_dim if self.tfidf_dim > 0 else None,
        }
        if override:
            params.update(override)
        return TfidfVectorizer(**params)

    def fit(self, texts: List[str]) -> "FeatureBuilder":
        eng = _engineered_df(texts)  # not used for fitting, but validates pipeline
        logger.info(f"Engineered features shape (fit phase): {eng.shape}")

        vec = self._make_vectorizer()
        if self.tfidf_dim == 0:
            self.vectorizer = vec
            return self

        try:
            vec.fit(texts)
        except ValueError:
            # Empty or degenerate corpus; fallback to char analyzer
            logger.warning("Word-level TF-IDF failed; switching to char 3-5.")
            vec = self._make_vectorizer({"analyzer": "char", "ngram_range": (3, 5)})
            vec.fit(texts)

        vocab_size = len(getattr(vec, "vocabulary_", {}))
        if vocab_size < max(50, int(self.tfidf_dim * 0.5)):
            logger.warning(
                f"TF-IDF vocab small ({vocab_size}); refitting with char 3-5."
            )
            vec = self._make_vectorizer({"analyzer": "char", "ngram_range": (3, 5)})
            vec.fit(texts)
            vocab_size = len(getattr(vec, "vocabulary_", {}))

        logger.info(f"TF-IDF fitted with vocab size: {vocab_size}")
        self.vectorizer = vec
        return self

    def transform(self, texts: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        eng = _engineered_df(texts)
        if self.tfidf_dim > 0:
            check_is_fitted(self.vectorizer, "vocabulary_")
            X = self.vectorizer.transform(texts)
            tf_names = [f"tfidf:{t}" for t in self.vectorizer.get_feature_names_out()]
            tf_df = pd.DataFrame(X.toarray(), columns=tf_names)
            df = pd.concat([eng, tf_df], axis=1)
            names = list(eng.columns) + tf_names
        else:
            df = eng
            names = list(eng.columns)

        logger.info(f"Final feature frame shape: {df.shape}")
        return df, names

    def fit_transform(self, texts: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        self.fit(texts)
        return self.transform(texts)

    def save_vectorizer(self, path: str) -> None:
        if self.vectorizer is not None:
            dump(self.vectorizer, path)
            logger.info(f"Saved vectorizer to: {path}")

def build_and_save_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: FeaturizerConfig,
    out_dir: str,
    out_train: str,
    out_val: str,
    out_test: str,
    names_json: str,
    vec_path: str,
    text_col: str = "text",
    label_col: str = "label",
) -> None:
    fb = FeatureBuilder(cfg)

    # Fit on train texts only
    train_texts = train_df[text_col].astype(str).tolist()
    val_texts   = val_df[text_col].astype(str).tolist()
    test_texts  = test_df[text_col].astype(str).tolist()

    Xtr, names = fb.fit_transform(train_texts)
    Xva, _ = fb.transform(val_texts)
    Xte, _ = fb.transform(test_texts)

    # Append labels
    Xtr[label_col] = train_df[label_col].to_numpy()
    Xva[label_col] = val_df[label_col].to_numpy()
    Xte[label_col] = test_df[label_col].to_numpy()

    # Save CSVs
    od = ensure_dir(out_dir)
    Xtr.to_csv(od / out_train, index=False)
    Xva.to_csv(od / out_val, index=False)
    Xte.to_csv(od / out_test, index=False)
    logger.info(f"Wrote features: {(od / out_train)} | {(od / out_val)} | {(od / out_test)}")

    # Save names and vectorizer
    with open(od / names_json, "w", encoding="utf-8") as f:
        json.dump(names + [label_col], f, ensure_ascii=False, indent=2)
    
    # Ensure parent directory for vectorizer exists before saving
    from pathlib import Path
    import shutil
    vec_p = Path(vec_path)
    if vec_p.is_dir():
        logger.warning(f"Output path {vec_p} is a directory, removing it.")
        shutil.rmtree(vec_p)
    vec_p.parent.mkdir(parents=True, exist_ok=True)
    
    fb.save_vectorizer(vec_path)
