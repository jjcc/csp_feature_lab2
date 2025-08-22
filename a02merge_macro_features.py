#!/usr/bin/env python3
# Complement VIX + per-symbol price-based features into labeled_trades_normal.csv

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import yfinance as yf

def _coerce_dt(s):
    return pd.to_datetime(s, errors="coerce")

def _load_vix(vix_csv, start_date, end_date, use_yf=True):
    # Return a Series indexed by date (daily), name="VIX"
    if vix_csv and Path(vix_csv).exists():
        vdf = pd.read_csv(vix_csv)
        date_col = "Date" if "Date" in vdf.columns else "date"
        close_col = "VIX" if "VIX" in vdf.columns else "Close"
        vdf[date_col] = pd.to_datetime(vdf[date_col], errors="coerce")
        vdf = vdf.dropna(subset=[date_col]).set_index(date_col).sort_index()
        return vdf.loc[start_date:end_date, close_col].rename("VIX").astype(float)
    #if use_yf:
    else:
        try:
            df = yf.download("^VIX", start=start_date, end=end_date)
            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Get single-index DataFrame by selecting the ^VIX ticker level
                vix_df = df.xs("^VIX", axis=1, level=1)  # Extract ^VIX columns, drop MultiIndex
                s = vix_df["Close"].rename("VIX")
            else:
                s = df["Close"].rename("VIX")
            s.index = pd.to_datetime(s.index)
            s.to_csv(vix_csv)
            return s
        except Exception:
            pass
    return pd.Series(dtype=float, name="VIX")

def _load_symbol_prices(symbol, px_dir, start_date, end_date, use_yf=True):
    # Return a Series indexed by date (business days ok), name="Close"
    if px_dir:
        #f = Path(px_dir) / f"{symbol}.csv"
        f = Path(px_dir) / f"{symbol}.parquet"
        if f.exists():
            #df = pd.read_csv(f)
            df = pd.read_parquet(f)
            # Check if index is already datetime, otherwise look for Date/date columns
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.sort_index()
                close_col = "Close" if "Close" in df.columns else "close"
                return df.loc[start_date:end_date, close_col].rename("Close").astype(float)
            else:
                date_col = "Date" if "Date" in df.columns else "date"
                close_col = "Close" if "Close" in df.columns else "close"
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
                return df.loc[start_date:end_date, close_col].rename("Close").astype(float)
    if use_yf and symbol:
        try:
            import yfinance as yf
            s = yf.download(symbol, start=start_date, end=end_date)
            s = s["Close"].rename("Close")
            s.index = pd.to_datetime(s.index)
            return s
        except Exception:
            pass
    return pd.Series(dtype=float, name="Close")

def _per_symbol_feature_frame(s_px):
    """
    Build per-symbol daily feature frame:
      ret_2d, ret_5d, ret_2d_norm, ret_5d_norm, prev_close
    All indexed by date (business days).
    """
    if s_px.empty:
        return pd.DataFrame()
    # Align to business days and forward-fill to avoid holes
    daily = s_px.asfreq("B").ffill()
    ret1d = daily.pct_change()
    df = pd.DataFrame(index=daily.index)
    df["prev_close"] = daily.shift(1)  # yesterday's close (no leakage)
    df["ret_2d"] = ret1d.rolling(2, min_periods=2).sum()
    df["ret_5d"] = ret1d.rolling(5, min_periods=5).sum()
    vol20 = ret1d.rolling(20, min_periods=5).std()
    df["ret_2d_norm"] = df["ret_2d"] / vol20.replace(0, np.nan)
    df["ret_5d_norm"] = df["ret_5d"] / vol20.replace(0, np.nan)
    return df

def main():
    load_dotenv()

    out_dir = os.getenv("OUT_DIR", "output")
    in_csv  = f"{out_dir}/labeled_trades_normal.csv"
    out_csv = f"{out_dir}/labeled_trades_with_macro.csv"

    # Sources
    VIX_CSV     = os.getenv("VIX_CSV", "").strip() or None
    PX_BASE_DIR = os.getenv("PX_BASE_DIR", "").strip() or None  # dir with <SYMBOL>.csv, Date, Close
    USE_YFIN    = bool(int(os.getenv("USE_YFINANCE", "1")))

    # Read trades
    df = pd.read_csv(in_csv)
    if "tradeTime" not in df.columns or "expirationDate" not in df.columns:
        raise SystemExit("Expected columns tradeTime and expirationDate in labeled_trades_normal.csv")

    # Parse dates and derive trade_date (calendar day)
    df["tradeTime"] = _coerce_dt(df["tradeTime"])
    df["expirationDate"] = _coerce_dt(df["expirationDate"])
    df["trade_date"] = df["tradeTime"].dt.floor("D")

    # VIX (global)
    start_date = df["trade_date"].min() # - pd.Timedelta(days=60)
    end_date   = df["trade_date"].max() + pd.Timedelta(days=1)
    vix = _load_vix(VIX_CSV,start_date, end_date, use_yf=USE_YFIN)
    vix_df = pd.DataFrame({"trade_date": vix.index, "VIX": vix.values})

    # Per-symbol price features
    need_symbol = "baseSymbol" in df.columns
    if not need_symbol:
        # Fallback to a single stream would be possible, but per your request,
        # we only do per-symbol merges here.
        raise SystemExit("Expected baseSymbol column for per-symbol price features.")

    # Compute DTE features on the row level (calendar days, clipped to ≥1)
    d = df.copy()
    d["daysToExpiration"] = ((d["expirationDate"].dt.floor("D") - d["tradeTime"].dt.floor("D")).dt.days).clip(lower=1)
    d["log1p_DTE"] = np.log1p(d["daysToExpiration"].astype(float))

    # Will collect per-symbol PX frames and merge back row-wise
    feats = []
    for sym in d["baseSymbol"].dropna().unique():
        s_mask = d["baseSymbol"] == sym
        date_span_min = d.loc[s_mask, "trade_date"].min() - pd.Timedelta(days=60)
        date_span_max = d.loc[s_mask, "trade_date"].max() + pd.Timedelta(days=1)

        s_px = _load_symbol_prices(sym, PX_BASE_DIR, date_span_min, date_span_max, use_yf=USE_YFIN)
        f = _per_symbol_feature_frame(s_px)
        if f.empty:
            # still create a frame with index to allow merge (will be NaN)
            f = pd.DataFrame(index=pd.date_range(date_span_min, date_span_max, freq="B"))
        f = f.reset_index().rename(columns={"index": "trade_date"})
        f["baseSymbol"] = sym
        feats.append(f)

    px_feat = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame(columns=["trade_date","baseSymbol"])
    # Merge VIX (by date) and PX features (by date+symbol)
    d = d.merge(vix_df, on="trade_date", how="left")
    # Fill missing VIX values with the previous date value of vix_df
    d["VIX"].fillna(method="ffill", inplace=True)

    px_feat.rename(columns={"Date": "trade_date"}, inplace=True)
    d = d.merge(px_feat, on=["trade_date","baseSymbol"], how="left")

    # Gap between previous close and current strike
    # (prev_close is yesterday’s close aligned to trade_date; strike is row-specific)
    if "strike" in d.columns:
        d["prev_close_minus_strike"] = d["prev_close"] - d["strike"]
        d["prev_close_minus_strike_pct"] = (d["prev_close"] - d["strike"]) / d["strike"].replace(0, np.nan)
    else:
        d["prev_close_minus_strike"] = np.nan
        d["prev_close_minus_strike_pct"] = np.nan

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
