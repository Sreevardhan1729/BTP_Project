#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# robust package import path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.fdof_pipeline.utils.io import get_logger
from src.fdof_pipeline.models.llm.prompts import (
    build_prompt_and_label, build_inference_prompt
)
from src.fdof_pipeline.models.llm.trainer import (
    TrainCfg, train_causal_lm, save_lines, predict_labels_greedy
)

logger = get_logger("fdof.train_llm")

def _load_selected(path: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in {path}")
    # Ensure numeric features and no NaNs
    feat_cols = [c for c in df.columns if c != label_col]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def main():
    parser = argparse.ArgumentParser(description="Step 7: LLM SFT on feature-informed prompts")
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
    trn = cfg["training"]
    gen = cfg["gen"]
    out = cfg["output"]

    label_col = inp.get("label_col", "label")
    train_df = _load_selected(inp["train_csv"], label_col)
    val_df   = _load_selected(inp["val_csv"], label_col)
    test_df  = _load_selected(inp["test_csv"], label_col)

    # --- Build SFT texts ---
    train_texts = []
    for _, row in train_df.iterrows():
        txt, _ = build_prompt_and_label(
            row, label_col,
            tfidf_prefix=prm.get("tfidf_prefix", "tfidf:"),
            top_k_tfidf=int(prm.get("top_k_tfidf", 20)),
            round_decimals=int(prm.get("round_decimals", 3)),
            include_engineered=bool(prm.get("include_engineered", True)),
        )
        train_texts.append(txt)

    val_texts = []
    for _, row in val_df.iterrows():
        txt, _ = build_prompt_and_label(
            row, label_col,
            tfidf_prefix=prm.get("tfidf_prefix", "tfidf:"),
            top_k_tfidf=int(prm.get("top_k_tfidf", 20)),
            round_decimals=int(prm.get("round_decimals", 3)),
            include_engineered=bool(prm.get("include_engineered", True)),
        )
        val_texts.append(txt)

    # Save SFT corpora for inspection
    save_lines(out["train_txt"], train_texts)
    save_lines(out["val_txt"], val_texts)

    # --- Train tiny causal LM (SFT-style) ---
    tcfg = TrainCfg(
        model_name_or_path=str(mdl.get("model_name_or_path", "sshleifer/tiny-gpt2")),
        max_length=int(mdl.get("max_length", 512)),
        seed=int(mdl.get("seed", 42)),
        epochs=int(trn.get("epochs", 1)),
        batch_size=int(trn.get("batch_size", 8)),
        learning_rate=float(trn.get("learning_rate", 5e-5)),
        weight_decay=float(trn.get("weight_decay", 0.0)),
        warmup_ratio=float(trn.get("warmup_ratio", 0.0)),
        gradient_accumulation_steps=int(trn.get("gradient_accumulation_steps", 1)),
        logging_steps=int(trn.get("logging_steps", 50)),
        save_total_limit=int(trn.get("save_total_limit", 1)),
        out_model_dir=str(out.get("model_dir", "outputs/artifacts/llm/model")),
    )

    model, tokenizer = train_causal_lm(train_texts, val_texts, tcfg)

    # --- Inference on test ---
    inf_prompts = []
    y_true = []
    for _, row in test_df.iterrows():
        inf_prompts.append(
            build_inference_prompt(
                row.drop(labels=[label_col]),
                tfidf_prefix=prm.get("tfidf_prefix", "tfidf:"),
                top_k_tfidf=int(prm.get("top_k_tfidf", 20)),
                round_decimals=int(prm.get("round_decimals", 3)),
                include_engineered=bool(prm.get("include_engineered", True)),
            )
        )
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
    logger.info(f"Saved model dir -> {out['model_dir']}")
    logger.info(f"Saved metrics -> {out['metrics_json']}")
    logger.info(f"Saved report  -> {out['report_txt']}")
    logger.info(f"Saved preds   -> {out['preds_csv']}")
    logger.info("LLM training complete.")

if __name__ == "__main__":
    main()
