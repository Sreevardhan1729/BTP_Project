#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import logging # Added for logging configuration

# robust package import path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.fdof_pipeline.utils.io import get_logger
from src.fdof_pipeline.models.llm.trainer import predict_labels_greedy

logger = get_logger("fdof.predict_llm_fewshot")

def _load_selected(path: str, label_col: str, text_col: str) -> pd.DataFrame:
    """Loads data, ensuring text and label cols are preserved."""
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in {path}")
    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not found in {path}")
    # Ensure numeric features and no NaNs
    feat_cols = [c for c in df.columns if c not in [label_col, text_col]]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def select_few_shot_examples(df: pd.DataFrame, k: int, strategy: str = "random") -> pd.DataFrame:
    """
    Selects k examples from a DataFrame based on a given strategy.
    
    Args:
        df: The DataFrame to sample from.
        k: The number of examples to select.
        strategy: The selection strategy ('random', etc.).
    
    Returns:
        A DataFrame containing the selected examples.
    """
    if strategy == "random":
        return df.sample(n=k)
    # In the future, you can add more strategies like:
    # elif strategy == "cosine_similarity":
    #     # Placeholder for a more complex selection logic
    #     pass
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")

def format_features(
    row: pd.Series,
    tfidf_prefix: str,
    top_k_tfidf: int,
    round_decimals: int,
    include_engineered: bool
) -> str:
    """Formats the feature part of a prompt for a single data row."""
    lines = []
    # TF-IDF features
    tfidf_cols = sorted([c for c in row.index if str(c).startswith(tfidf_prefix)],
                        key=lambda c: row[c], reverse=True)
    if len(tfidf_cols) > 0:
        top_tfidf = tfidf_cols[:top_k_tfidf]
        tfidf_str = ", ".join([
            f"{c.replace(tfidf_prefix, '')} ({row[c]:.{round_decimals}f})" for c in top_tfidf
        ])
        lines.append(f"Key Terms: {tfidf_str}")

    # Engineered features
    if include_engineered:
        eng_cols = [c for c in row.index if not str(c).startswith(tfidf_prefix)]
        eng_str = ", ".join([f"{c} ({row[c]:.{round_decimals}f})" for c in eng_cols])
        lines.append(f"Textual Features: {eng_str}")
    
    return "\n".join(lines)

def build_few_shot_prompt(
    test_row: pd.Series,
    examples_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    prm: dict,
) -> str:
    """Builds a few-shot prompt with examples and a final query."""
    prompt = "Analyze the text and its features to classify it. Here are some examples:\n"
    
    # Add examples to the prompt
    for _, ex_row in examples_df.iterrows():
        features_str = format_features(
            ex_row.drop(labels=[label_col, text_col]),
            tfidf_prefix=prm.get("tfidf_prefix", "tfidf:"),
            top_k_tfidf=int(prm.get("top_k_tfidf", 20)),
            round_decimals=int(prm.get("round_decimals", 3)),
            include_engineered=bool(prm.get("include_engineered", True)),
        )
        prompt += (
            "\n--- Example ---\n"
            f"Text: {ex_row[text_col]}\n"
            f"Features:\n{features_str}\n"
            f"Label: {int(ex_row[label_col])}\n"
        )
    
    # Add the final query for the test instance
    test_features_str = format_features(
        test_row.drop(labels=[label_col, text_col]),
        tfidf_prefix=prm.get("tfidf_prefix", "tfidf:"),
        top_k_tfidf=int(prm.get("top_k_tfidf", 20)),
        round_decimals=int(prm.get("round_decimals", 3)),
        include_engineered=bool(prm.get("include_engineered", True)),
    )
    prompt += (
        "\n--- Task ---\n"
        f"Text: {test_row[text_col]}\n"
        f"Features:\n{test_features_str}\n"
        "Label:"
    )
    return prompt

def main():
    # Suppress transformers library warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    parser = argparse.ArgumentParser(description="Step 7: LLM prediction with in-context learning")
    parser.add_argument("--config", type=str, default="configs/config-llm.yaml",
                        help="Path to LLM config YAML")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inp = cfg["input"]
    prm = cfg["prompt"]
    mdl = cfg["model"]
    gen = cfg["gen"]
    out = cfg["output"]
    icl = cfg.get("in_context_learning", {"num_examples": 3, "strategy": "random"})

    # --- Create unique output paths based on model name ---
    model_name = str(mdl.get("model_name_or_path", "sshleifer/tiny-gpt2"))
    model_fname = model_name.replace("/", "_")
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    out["metrics_json"] = str(Path(out["metrics_json"]).parent / f"llm_{model_fname}_{ts}_metrics.json")
    out["report_txt"] = str(Path(out["report_txt"]).parent / f"llm_{model_fname}_{ts}_report.txt")
    out["preds_csv"] = str(Path(out["preds_csv"]).parent / f"llm_{model_fname}_{ts}_preds.csv")

    label_col = inp.get("label_col", "label")
    text_col = inp.get("text_col", "text") # Assumes a 'text' column in your CSVs
    train_df = _load_selected(inp["train_csv"], label_col, text_col)
    test_df  = _load_selected(inp["test_csv"], label_col, text_col)

    # --- Load pre-trained model for inference ---
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    logger.info("Model loaded successfully.")

    # --- Inference on test ---
    inf_prompts = []
    y_true = []
    num_examples = icl.get("num_examples", 3)
    strategy = icl.get("strategy", "random")

    for _, row in test_df.iterrows():
        examples = select_few_shot_examples(train_df, k=num_examples, strategy=strategy)
        prompt = build_few_shot_prompt(row, examples, text_col, label_col, prm)
        inf_prompts.append(prompt)
        y_true.append(int(row[label_col]))

    y_pred = predict_labels_greedy(
        model=model, tokenizer=tokenizer,
        prompts=inf_prompts,
        max_new_tokens=int(gen.get("max_new_tokens", 1)),
    )

    # --- Metrics & save ---
    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1": float(f1_score(y_true, y_pred)),
        "test_precision": float(precision_score(y_true, y_pred)),
        "test_recall": float(recall_score(y_true, y_pred)),
    }
    Path(out["metrics_json"]).parent.mkdir(parents=True, exist_ok=True)
    with open(out["metrics_json"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(y_true, y_pred, digits=4)
    Path(out["report_txt"]).parent.mkdir(parents=True, exist_ok=True)
    with open(out["report_txt"], "w", encoding="utf-8") as f:
        f.write(report + "\n")

    Path(out["preds_csv"]).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(out["preds_csv"], index=False)

    logger.info(f"LLM test metrics: {metrics}")
    logger.info(f"Saved metrics -> {out['metrics_json']}")
    logger.info(f"Saved report  -> {out['report_txt']}")
    logger.info(f"Saved preds   -> {out['preds_csv']}")
    logger.info("LLM prediction complete.")

if __name__ == "__main__":
    main()
