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

def _load_vix(vix_csv, start_date, end_date):
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

def _load_symbol_prices(symbol, px_dir, start_date, end_date, use_yf=False):
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
        raise NotImplementedError("Yahoo Finance integration not implemented")
        try:
            import yfinance as yf
            s = yf.download(symbol, start=start_date, end=end_date)
            s = s["Close"].rename("Close")
            s.index = pd.to_datetime(s.index)
            return s
        except Exception:
            pass
    return pd.Series(dtype=float, name="Close")

def _per_symbol_feature_frame(s_px: pd.Series, start_date, max_trade_date) -> pd.DataFrame:
    """
    Build per-symbol daily features with a row for every business day up to max_trade_date,
    but never fabricate today's (T) close if it doesn't exist. No leakage:
      prev_close(T)      = Close(T-1)
      ret_2d(T), ret_5d(T) use returns up to T-1
    """
    import numpy as np
    import pandas as pd

    if s_px.empty:
        # still return an empty frame on the requested calendar so merge works
        cal = pd.bdate_range(start=start_date, end=max_trade_date)
        return pd.DataFrame(index=cal)

    s_px = s_px.sort_index()
    last_px_date = s_px.index.max()

    # Calendar up to trade-date T (even if T's close is missing)
    cal = pd.bdate_range(start=start_date, end=max_trade_date)

    # Reindex to calendar; ffill only up to the last REAL close
    daily = s_px.reindex(cal)
    daily_ff = daily.ffill()
    # critical line: DO NOT ffill beyond last_px_date (keeps today's Close as NaN)
    daily_ff.loc[daily_ff.index > last_px_date] = np.nan

    # 1-day returns on ffilled series; today's ret is NaN if today's Close is NaN
    ret1d = daily_ff.pct_change()
    # strict lag so all rolling windows end at T-1
    ret1d_lag = ret1d.shift(1)

    out = pd.DataFrame(index=cal)
    out["prev_close"]   = daily_ff.shift(1)                     # Close(T-1)
    out["ret_2d"]       = ret1d_lag.rolling(2, min_periods=2).sum()
    out["ret_5d"]       = ret1d_lag.rolling(5, min_periods=5).sum()
    vol20               = ret1d_lag.rolling(20, min_periods=5).std()
    out["ret_2d_norm"]  = out["ret_2d"] / vol20.replace(0, np.nan)
    out["ret_5d_norm"]  = out["ret_5d"] / vol20.replace(0, np.nan)
    return out

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

    # Parse dates and derive trade_date (calendar day)
    df["tradeTime"] = _coerce_dt(df["tradeTime"])
    df["expirationDate"] = _coerce_dt(df["expirationDate"])
    df["trade_date"] = df["tradeTime"].dt.floor("D")

    # VIX (global)
    start_date = df["trade_date"].min() # - pd.Timedelta(days=60)
    end_date   = df["trade_date"].max() + pd.Timedelta(days=1)
    vix = _load_vix(VIX_CSV,start_date, end_date)
    vix_df = pd.DataFrame({"trade_date": vix.index, "VIX": vix.values})

    # Per-symbol price features
    need_symbol = "baseSymbol" in df.columns
    d = per_symbol_price_feat(PX_BASE_DIR, df, vix_df, need_symbol)

    # Gap between previous close and current underlying price
    if "underlyingLastPrice" in d.columns:
        d["prev_close_minus_ul"] = d["prev_close"] - d["underlyingLastPrice"]
        d["prev_close_minus_ul_pct"] = (d["prev_close"] - d["underlyingLastPrice"]) / d["underlyingLastPrice"].replace(0, np.nan)
    else:
        d["prev_close_minus_ul"] = np.nan
        d["prev_close_minus_ul_pct"] = np.nan

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

def per_symbol_price_feat(PX_BASE_DIR, df, vix_df, need_symbol):
    if not need_symbol:
        # Fallback to a single stream would be possible, but per your request,
        # we only do per-symbol merges here.
        raise SystemExit("Expected baseSymbol column for per-symbol price features.")

    # Compute DTE features on the row level (calendar days, clipped to ≥1)
    d = df.copy()

    d["log1p_DTE"] = np.log1p(d["daysToExpiration"].astype(float))

    # Will collect per-symbol PX frames and merge back row-wise
    feats = []
    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce").dt.floor("D")
    for sym in d["baseSymbol"].dropna().unique():
        s_mask = d["baseSymbol"] == sym
        start = d.loc[s_mask, "trade_date"].min() - pd.Timedelta(days=60)
        # IMPORTANT: end at the **max trade_date** (T), not +1BD in real time
        stop  = d.loc[s_mask, "trade_date"].max()

        s_px = _load_symbol_prices(sym, PX_BASE_DIR, start, stop)
        f = _per_symbol_feature_frame(s_px, start, stop)
        if f.empty:
            f = pd.DataFrame(index=pd.date_range(start, stop, freq="B"))

        f = f.reset_index().rename(columns={"index": "trade_date"})
        f["baseSymbol"] = sym
        feats.append(f)

    px_feat = (pd.concat(feats, ignore_index=True)
               if feats else pd.DataFrame(columns=["trade_date", "baseSymbol"]))

    # Merge VIX by date (then ffill gaps)
    d = d.merge(vix_df, on="trade_date", how="left")
    d["VIX"] = d["VIX"].ffill()

    # Merge per-date, per-symbol PX features — gives different values per trade date
    d = d.merge(px_feat, on=["trade_date","baseSymbol"], how="left")
    return d

if __name__ == "__main__":
    main()
