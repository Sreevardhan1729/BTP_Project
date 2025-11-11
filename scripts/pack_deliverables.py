#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import glob
import zipfile
import yaml
from datetime import datetime

def _expand_patterns(base_dir: Path, patterns: list[str]) -> list[Path]:
    files = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            path = Path(p)
            if path.is_file():
                files.append(path)
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for f in files:
        if f.resolve() not in seen:
            uniq.append(f)
            seen.add(f.resolve())
    return uniq

def main():
    parser = argparse.ArgumentParser(description="Step 9: Bundle deliverables into a ZIP")
    parser.add_argument("--config", type=str, default="configs/config-run.yaml",
                        help="Path to run config YAML (for pack section)")
    parser.add_argument("--zip", type=str, default=None,
                        help="Override output zip path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f)

    pack_cfg = run_cfg.get("pack", {})
    out_zip = Path(args.zip or pack_cfg.get("output_zip", "outputs/deliverables/fdof_deliverable.zip"))
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    include_patterns = pack_cfg.get("include", [])
    files = _expand_patterns(Path("."), include_patterns)

    # If zip already exists, timestamp it to avoid IsADirectoryError/overwrite confusion
    if out_zip.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_zip = out_zip.with_name(out_zip.stem + f"_{ts}" + out_zip.suffix)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.as_posix())
    print(f"[INFO] Wrote deliverable -> {out_zip} ({len(files)} files)")

if __name__ == "__main__":
    main()
