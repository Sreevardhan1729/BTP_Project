#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.fdof_pipeline.data.load import load_and_standardize
from src.fdof_pipeline.data.split import stratified_train_val
from src.fdof_pipeline.utils.io import ensure_dir, get_logger

logger = get_logger("fdof.prepare")

def main():
    parser = argparse.ArgumentParser(description="Step 1: Environment & Data setup")
    parser.add_argument("--config", type=str, default="configs/config-data.yaml", help="Path to data config YAML")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inp = cfg["input"]
    out = cfg["output"]
    split_cfg = cfg.get("split", {})
    val_size = float(split_cfg.get("val_size", 0.2))
    seed = int(split_cfg.get("seed", 42))

    # 1) Load & standardize
    train_df, test_df = load_and_standardize(
        train_csv=inp["train_csv"],
        test_csv=inp["test_csv"],
        text_col=inp["text_col"],
        label_col=inp["label_col"],
        drop_cols=inp.get("drop_cols", []),
    )

    # 2) Make stratified validation split from TRAIN
    train_df, val_df = stratified_train_val(train_df, val_size=val_size, seed=seed)

    # 3) Write standardized CSVs
    out_dir = ensure_dir(out["processed_dir"])
    train_out = out_dir / out["train_out"]
    val_out = out_dir / out["val_out"]
    test_out = out_dir / out["test_out"]

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    logger.info(f"Wrote: {train_out}")
    logger.info(f"Wrote: {val_out}")
    logger.info(f"Wrote: {test_out}")
    logger.info("Done.")

if __name__ == "__main__":
    main()
