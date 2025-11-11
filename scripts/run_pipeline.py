#!/usr/bin/env python
#python scripts/run_pipeline.py --config configs/config-run.yaml``
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import yaml

def run_step(name: str, script: str, config_path: str) -> None:
    print(f"[RUN] {name}: {script} --config {config_path}")
    cmd = [sys.executable, script, "--config", config_path]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Step 9: End-to-end pipeline runner")
    parser.add_argument("--config", type=str, default="configs/config-run.yaml",
                        help="Path to run config YAML")
    parser.add_argument("--start", type=str, default=None,
                        help="Start from this step (inclusive).")
    parser.add_argument("--end", type=str, default=None,
                        help="End at this step (inclusive).")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Run config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    steps = cfg.get("steps", {})
    # Preserve insertion order (PyYAML keeps it)
    names = list(steps.keys())

    # Apply start/end window if provided
    if args.start:
        if args.start not in steps:
            raise KeyError(f"--start '{args.start}' not in steps: {names}")
        names = names[names.index(args.start):]
    if args.end:
        if args.end not in steps:
            raise KeyError(f"--end '{args.end}' not in steps: {names}")
        names = names[:names.index(args.end)+1]

    for name in names:
        s = steps[name]
        if not bool(s.get("enabled", True)):
            print(f"[SKIP] {name}")
            continue
        script = s["script"]
        config_path = s["config"]
        run_step(name, script, config_path)

    print("[DONE] Pipeline completed.")

if __name__ == "__main__":
    main()
