#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd
import yaml
from datetime import datetime

def _maybe_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _maybe_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def _section(title: str) -> str:
    return f"# {title}\n\n"

def _kv_table(d: dict) -> str:
    if not d:
        return "_N/A_\n\n"
    lines = ["| Key | Value |", "|---|---|"]
    for k, v in d.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n\n"

def _df_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_N/A_\n\n"
    if len(df) > max_rows:
        df = df.head(max_rows).copy()
    # round floats for readability
    for c in df.select_dtypes(include="number").columns:
        df[c] = df[c].round(4)
    return df.to_markdown(index=False) + "\n\n"

def main():
    parser = argparse.ArgumentParser(description="Step 9: Generate final Markdown report")
    parser.add_argument("--config", type=str, default="configs/config-run.yaml",
                        help="Path to run config YAML (for output path)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f)

    out_md = Path(run_cfg.get("report", {}).get("output_markdown", "reports/final_report.md"))
    out_md.parent.mkdir(parents=True, exist_ok=True)

    # Gather sources
    svm_metrics = _maybe_json("outputs/results/svm_metrics.json")
    llm_metrics = _maybe_json("outputs/results/llm_metrics.json")
    outlier_tuning = _maybe_json("outputs/artifacts/outlier_cv_tuning.json")
    efsa_report = _maybe_json("outputs/artifacts/efsa_report.json")
    exp_summary = _maybe_csv("reports/tables/experiments_summary.csv")

    # Figures (optional – embed if present)
    figs = {
        "EFSA Progress": "reports/figures/efsa_progress.png",
        "Outlier Trials: k vs Acc": "reports/figures/outlier_trials_k_vs_acc.png",
        "Outlier Trials: Acc by Combine": "reports/figures/outlier_trials_acc_by_combine.png",
        "SVM Top Positive Features": "reports/figures/svm_top_positive_features.png",
        "SVM Top Negative Features": "reports/figures/svm_top_negative_features.png",
        "SVM Confusion Matrix": "reports/figures/svm_confusion_matrix.png",
    }

    # Compose markdown
    md = []
    md.append(_section("FDOF Pipeline — Final Report"))
    md.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")

    md.append(_section("1. Cross-validated Outlier Tuning (Step 4)"))
    best = (outlier_tuning or {}).get("best_params", {})
    scores = (outlier_tuning or {}).get("best_scores", {})
    md.append("**Best parameters**\n\n")
    md.append(_kv_table(best))
    md.append("**CV scores**\n\n")
    md.append(_kv_table(scores))

    md.append(_section("2. EFSA Feature Selection (Step 5)"))
    md.append(_kv_table(efsa_report or {}))

    md.append(_section("3. SVM Results (Step 6)"))
    md.append(_kv_table(svm_metrics or {}))

    md.append(_section("4. LLM Results (Step 7)"))
    md.append(_kv_table(llm_metrics or {}))

    md.append(_section("5. Comparative Experiments (Step 8)"))
    md.append(_df_table(exp_summary))

    md.append(_section("6. Key Figures"))
    for title, rel in figs.items():
        if Path(rel).exists():
            md.append(f"**{title}**\n\n![]({rel})\n\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("".join(md))

    print(f"[INFO] Wrote final report -> {out_md}")

if __name__ == "__main__":
    main()
