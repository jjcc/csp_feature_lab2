#!/usr/bin/env python3
# Complement VIX + per-symbol price-based features into labeled_trades_normal.csv

import os
import json
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

from service.data_prepare import add_macro_features



def main():
    load_dotenv()

    out_dir = os.getenv("OUT_DIR", "output")
    IN_CSV_FILE = os.getenv("GEX_CSV", "labeled_trades_gex_normal.csv")
    in_csv  = f"{out_dir}/{IN_CSV_FILE}"
    MACROFEATURE_CSV = os.getenv("MACRO_FEATURE_CSV", "labeled_trades_gex_macro.csv")   
    out_csv = f"{out_dir}/{MACROFEATURE_CSV}"

    # Sources
    VIX_CSV     = os.getenv("VIX_CSV", "").strip() or None
    PX_BASE_DIR = os.getenv("PX_BASE_DIR", "").strip() or None  # dir with <SYMBOL>.csv, Date, Close
    #USE_YFIN    = bool(int(os.getenv("USE_YFINANCE", "1")))

    # Read trades
    df = pd.read_csv(in_csv)
    if "tradeTime" not in df.columns or "expirationDate" not in df.columns:
        raise SystemExit("Expected columns tradeTime and expirationDate in labeled_trades_normal.csv")

    # Use shared macro features function
    d = add_macro_features(df, VIX_CSV, PX_BASE_DIR)

    # Save
    d.to_csv(out_csv, index=False)

    # Simple report
    rep = {
        "rows_in": int(len(df)),
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
