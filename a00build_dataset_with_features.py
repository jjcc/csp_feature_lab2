
#!/usr/bin/env python3
# Merge GEX features into  raw data defined in BASIC_CSV
# Now also includes macro features like VIX and price returns
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from datetime import time
from pathlib import Path
from service.data_prepare import add_macro_features
from service.preprocess import  load_csp_files, merge_gex
from service.env_config import getenv


def parse_target_time(s: str) -> time:
    try:
        hh, mm = s.split(":")
        return time(int(hh), int(mm))
    except Exception:
        return time(11, 0)



def main():
    data_dir = getenv("COMMON_DATA_DIR", "")
    glob_pat = getenv("DATA_GLOB", "coveredPut_*.csv")
    target_time = getenv("DATA_TARGET_TIME", "11:00")

    # inputs
    out_dir = getenv("COMMON_OUTPUT_DIR", "output")

    # GEX source
    base_dir = getenv("GEX_BASE_DIR")
    target_time_str = getenv("GEX_TARGET_TIME", "11:00")
    if not base_dir:
        raise SystemExit("GEX_BASE_DIR is not set in .env")

    # VIX and price sources
    VIX_CSV     = getenv("MACRO_VIX_CSV", "").strip() or None
    PX_BASE_DIR = getenv("MACRO_PX_BASE_DIR", "").strip() or None  # dir with <SYMBOL>.csv, Date, Close

    # output
    MACROFEATURE_CSV = getenv("COMMON_MACRO_FEATURE_CSV", "labeled_trades_gex_macro.csv")
    out_csv = f"{out_dir}/{MACROFEATURE_CSV}"

    target_t = parse_target_time(target_time_str)
    target_minutes = target_t.hour * 60 + target_t.minute

    # Step 1: Load raw data from multiple files
    raw = load_csp_files(data_dir, glob_pat, target_time=target_time, enforce_daily_pick=True)
    #raw = load_csp_files(data_dir, glob_pat, target_time=target_time, enforce_daily_pick=False)
    raw = raw.drop(columns=["baseSymbolType","Unnamed: 0", "symbolType"], errors='ignore')
    # rename index to "row_id" for tracking
    raw = raw.reset_index().rename(columns={"index": "row_id"})
    # raw is not written but used directly below
    #trades = pd.read_csv(csv_path)
    trades = raw

    # Step 2: Merge GEX features
    gex_merged = merge_gex(trades, base_dir, target_minutes)

    # Step 3: Add macro features , use shared macro features function
    d = add_macro_features(gex_merged, VIX_CSV, PX_BASE_DIR)
    # filter rows with missing GEX if specified. Default: keep all rows
    if getenv("GEX_FILTER", "0").strip() in {"1","true","yes","y","on"}:
        d = d[d["gex_missing"] == 0].copy()
        out_csv = out_csv.replace(".csv", "_gexonly.csv")
        print(f"Filtered rows with missing GEX, remaining {len(d)} rows.")

    # Save
    d.to_csv(out_csv, index=False)

    # Simple report
    rep = {
         "rows_gex_merged": len(gex_merged),
        "gex_found": int((gex_merged["gex_missing"] == 0).sum()),
        "gex_missing": int((gex_merged["gex_missing"] == 1).sum()),
        "base_dir": base_dir,
        "target_time": target_time_str,
        "rows_out": int(len(d)),
        "unique_symbols": int(d["baseSymbol"].nunique()) if "baseSymbol" in d.columns else None,
        "vix_non_null": int(d["VIX"].notna().sum()),
        "prev_close_non_null": int(d["prev_close"].notna().sum()),
        "px_base_dir": PX_BASE_DIR,
        "out_csv": out_csv
    }
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
