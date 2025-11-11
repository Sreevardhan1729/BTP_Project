from __future__ import annotations
from typing import List, Tuple
import pandas as pd

try:
    # Prefer relative import to match your package style
    from ...features.build_features import ENGINEERED_ORDER as _ENGINEERED_ORDER
except Exception:
    # Fallback if import path changes; engineered set can be empty
    _ENGINEERED_ORDER = []

ENGINEERED_SET = set(_ENGINEERED_ORDER)

TEMPLATE_INSTRUCTION = (
    "You are a misinformation classifier. Use the signals to output a single digit: "
    "0 for fake/misleading, 1 for real/credible. Reply with just the digit."
)

def _row_signals(
    row: pd.Series,
    tfidf_prefix: str = "tfidf:",
    top_k_tfidf: int = 20,
    round_decimals: int = 3,
    include_engineered: bool = True,
) -> str:
    # Engineered features present in this row (intersect with selected columns)
    eng_pairs = []
    if include_engineered:
        for name in row.index:
            if name in ENGINEERED_SET:
                val = row[name]
                eng_pairs.append(f"{name}={round(float(val), round_decimals)}")
    eng_part = ""
    if eng_pairs:
        eng_part = "Engineered: " + ", ".join(eng_pairs)

    # Top-K TF-IDF features (by value) for this row
    tfidf_cols = [c for c in row.index if c.startswith(tfidf_prefix)]
    tfidf_vals = [(c, float(row[c])) for c in tfidf_cols if float(row[c]) > 0.0]
    tfidf_vals.sort(key=lambda x: x[1], reverse=True)
    tfidf_top = tfidf_vals[: max(0, int(top_k_tfidf))]

    tfidf_pairs = []
    for name, val in tfidf_top:
        tok = name.split(":", 1)[-1]
        tfidf_pairs.append(f"{tok}:{round(val, round_decimals)}")

    tfidf_part = ""
    if tfidf_pairs:
        tfidf_part = "Top-TFIDF: " + ", ".join(tfidf_pairs)

    parts = [p for p in [eng_part, tfidf_part] if p]
    return "\n".join(parts) if parts else "No signals."

def build_prompt_and_label(
    row: pd.Series,
    label_col: str,
    tfidf_prefix: str,
    top_k_tfidf: int,
    round_decimals: int,
    include_engineered: bool,
) -> Tuple[str, str]:
    label = str(int(row[label_col]))
    feat_row = row.drop(labels=[label_col])
    signals = _row_signals(
        feat_row,
        tfidf_prefix=tfidf_prefix,
        top_k_tfidf=top_k_tfidf,
        round_decimals=round_decimals,
        include_engineered=include_engineered,
    )
    # Prompt for SFT: we’ll train the model to produce the digit after "Answer:"
    prompt = f"{TEMPLATE_INSTRUCTION}\n\nSIGNALS:\n{signals}\n\nAnswer: {label}"
    return prompt, label

def build_inference_prompt(
    row: pd.Series,
    tfidf_prefix: str,
    top_k_tfidf: int,
    round_decimals: int,
    include_engineered: bool,
) -> str:
    feat_row = row
    signals = _row_signals(
        feat_row,
        tfidf_prefix=tfidf_prefix,
        top_k_tfidf=top_k_tfidf,
        round_decimals=round_decimals,
        include_engineered=include_engineered,
    )
    return f"{TEMPLATE_INSTRUCTION}\n\nSIGNALS:\n{signals}\n\nAnswer:"
