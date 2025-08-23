
#!/usr/bin/env python3
"""
CSP Feature Lab
---------------
Iterate all CSP snapshot CSVs in a folder, label outcomes at expiry, and run a
RandomForest feature-importance analysis to see which parameters relate to profitability.

Usage:
    pip install -r requirements.txt
    python analyze_csp_features.py --data_dir /path/to/folder --glob "coveredPut_*.csv" --max_rows 0

Outputs (in the same folder as this script):
    labeled_trades.csv        - all labeled rows (one per option row that could be labeled)
    feature_importances.csv   - sorted feature importances from RandomForest
    classification_report.txt - train/test metrics
"""

import os
from os import getenv
from pathlib import Path
import glob
import argparse
from time import sleep
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
from service.preprocess import load_csp_files 

from dotenv import load_dotenv;
load_dotenv()

def ensure_cache_dir(out_dir):
    pc = os.path.join(out_dir, "price_cache")
    os.makedirs(pc, exist_ok=True)
    return pc

def load_cached_price_data(cache_dir, symbol):
    p = os.path.join(cache_dir, f"{symbol}.parquet")
    if os.path.exists(p):
        try:
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"[WARN] cache read failed for {symbol}: {e}")
    return None

def save_cached_price_data(cache_dir, symbol, price_df: pd.DataFrame):
    try:
        price_df.to_parquet(os.path.join(cache_dir, f"{symbol}.parquet"))
    except Exception as e:
        print(f"[WARN] cache write failed for {symbol}: {e}")

def download_prices_batched(symbols, start_dt, end_dt, batch_size=30, threads=True):
    """
    Download daily OHLCV prices for many symbols in batches using yfinance.
    Returns dict: symbol -> pd.DataFrame (Open, High, Low, Close, Volume, etc.)
    """
    import yfinance as yf
    import math
    symbols = sorted(set([s for s in symbols if isinstance(s, str) and len(s)>0]))
    symbols = [s if "." not in s else s.replace(".", "_") for s in symbols]  # ensure uppercase
    out = {}
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            df = yf.download(batch, start=start_dt.date(), end=(end_dt + pd.Timedelta(days=1)).date(),
                             interval="1d", group_by="ticker", auto_adjust=False, threads=threads, progress=False)
            # yfinance returns multi-index columns when multiple tickers
            if isinstance(df.columns, pd.MultiIndex):
                # df columns like ('AAPL','Open'), ('AAPL','High'), ('AAPL','Low'), ('AAPL','Close'), etc.
                for sym in batch:
                    sym_cols = [col for col in df.columns if col[0] == sym]
                    if sym_cols:
                        sym_df = df[[col for col in sym_cols]]
                        # Flatten column names: ('AAPL', 'Close') -> 'Close'
                        sym_df.columns = [col[1] for col in sym_df.columns]
                        sym_df = sym_df.dropna(how='all')
                        sym_df.index = pd.to_datetime(sym_df.index)
                        out[sym] = sym_df
            else:
                # single ticker case - df already has columns like 'Open', 'High', 'Low', 'Close', etc.
                df = df.dropna(how='all')
                df.index = pd.to_datetime(df.index)
                out[batch[0]] = df
        except Exception as e:
            print(f"[WARN] batch download failed for {batch}: {e}")
        sleep(1)  # avoid hitting API limits
    return out

def preload_prices_with_cache(raw_df, out_dir, batch_size=30, cut_off_date=None):
    """
    From the raw CSP rows, determine unique symbols and date window,
    load from cache if available, batch-download missing ones, and save to cache.
    Returns dict: symbol -> DataFrame with OHLCV data
    """
    cache_dir = ensure_cache_dir(out_dir)
    # Determine symbols and window
    syms = raw_df['baseSymbol'].dropna().astype(str).str.upper().unique().tolist()
    # Window: from min(tradeTime, expiry)-5 days to max(expiry)+1 day
    tt = pd.to_datetime(raw_df.get('tradeTime', pd.NaT), errors="coerce")
    ed = pd.to_datetime(raw_df.get('expirationDate', pd.NaT), errors="coerce")
    start_dt = pd.to_datetime(min([d for d in pd.concat([tt, ed.dropna()]) if pd.notna(d)]) - pd.Timedelta(days=5))
    # check week day of start_dt
    if start_dt.weekday() == 5:  # Saturday
        start_dt += pd.Timedelta(days=2)
    elif start_dt.weekday() == 6:  # Sunday
        start_dt += pd.Timedelta(days=1)
    end_dt = pd.to_datetime(max([d for d in ed.dropna()]) + pd.Timedelta(days=1))
    if cut_off_date is  None:
        cut_off_date = pd.Timestamp.now()
    if end_dt > cut_off_date:
        end_dt = cut_off_date
        end_dt = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    # check week day of end_dt
    if end_dt.weekday() == 5:  # Saturday
        end_dt -= pd.Timedelta(days=1)
    elif end_dt.weekday() == 6:  # Sunday
        end_dt -= pd.Timedelta(days=2)
    # Load from cache or mark missing
    prices = {}
    missing = []
    for s in syms:
        price_df = load_cached_price_data(cache_dir, s)
        if price_df is not None and (price_df.index.min() <= start_dt) and (price_df.index.max() >= end_dt):
            prices[s] = price_df
        else:
            missing.append(s)
    if missing:
        print(f"[INFO] Downloading {len(missing)} symbols in batches (size={batch_size})...")
        fetched = download_prices_batched(missing, start_dt, end_dt, batch_size=batch_size, threads=True)
        for s, price_df in fetched.items():
            prices[s] = price_df
            save_cached_price_data(cache_dir, s, price_df)
    return prices

def lookup_close_on_or_before(price_df: pd.DataFrame, target_dt: pd.Timestamp) -> float:
    if price_df is None or price_df.empty or pd.isna(target_dt) or 'Close' not in price_df.columns:
        return np.nan
    sub = price_df[price_df.index<=pd.to_datetime(target_dt)]
    if len(sub)==0:
        return np.nan
    return float(sub['Close'].iloc[-1])


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan




def load_csp_files_old(data_dir: str, pattern: str) -> pd.DataFrame:
    """
    Concatenate all CSV files matching the pattern. Adds __source_file column.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__source_file"] = os.path.basename(p)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
    if not frames:
        raise SystemExit(f"No files found for pattern {pattern} in {data_dir}")
    return pd.concat(frames, ignore_index=True)  # (unused now)

def derive_capital(df, policy="strike100", constant_capital=10_000.0):
    strike = pd.to_numeric(df["strike"], errors="coerce")
    ul = pd.to_numeric(df.get("underlyingLastPrice"), errors="coerce")
    # per-share option price at entry (fallback to bidPrice)
    opt_px = pd.to_numeric(df.get("bid", df.get("bidPrice")), errors="coerce")
    entry_credit = pd.to_numeric(df["entry_credit"], errors="coerce")  # already *100

    if policy == "constant":
        df["capital"] = float(constant_capital)
        cap = df["capital"]
    elif policy == "strike100":
        cap = 100.0 * strike
    elif policy == "credit_adjusted":
        cap = 100.0 * strike - entry_credit
    elif policy == "regt_light":
        # Rough Reg-T proxy for short puts: per-share margin = option_px + max(0.20*UL - OTM, 0.10*UL)
        otm = np.maximum(ul - strike, 0.0)
        per_share_margin = opt_px + np.maximum(0.20*ul - otm, 0.10*ul)
        cap = 100.0 * per_share_margin
    else:
        cap = 100.0 * strike

    cap = pd.to_numeric(cap, errors="coerce").replace([np.inf,-np.inf], np.nan)
    # sensible floor to avoid divide-by-zero
    return pd.Series(cap).fillna(100.0 * strike).clip(lower=50.0)


def build_dataset(raw: pd.DataFrame, max_rows: int = 0, preload_closes: dict = None) -> pd.DataFrame:
    """
    Prepare labeled dataset for modeling.
    Assumes columns (case-sensitive): 
      baseSymbol, expirationDate, strike, bid, ask, delta, moneyness, impliedVolatilityRank1y, 
      potentialReturn, potentialReturnAnnual, breakEvenProbability, percentToBreakEvenBid,
      openInterest, volume, tradeTime, underlyingLastPrice
    Missing columns are tolerated and filled with NaN.
    """
    df = raw.copy()
    # Standardize expected columns
    expected_cols = [
        "baseSymbol","expirationDate","strike","bid","ask","delta","moneyness",
        "impliedVolatilityRank1y","potentialReturn","potentialReturnAnnual",
        "breakEvenProbability","percentToBreakEvenBid","openInterest","volume",
        "tradeTime","underlyingLastPrice","__source_file"
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Parse datetimes
    df["tradeTime"] = pd.to_datetime(df["tradeTime"], errors="coerce")
    df["expirationDate"] = pd.to_datetime(df["expirationDate"], errors="coerce")

    # Limit rows for a quick run if requested
    if max_rows and max_rows > 0:
        df = df.head(max_rows).copy()

    # Use preloaded prices to compute expiry_close
    def expiry_close_from_cache(r):
        if preload_closes is None:
            return np.nan
        sym = str(r['baseSymbol']).upper()
        price_df = preload_closes.get(sym)
        return lookup_close_on_or_before(price_df, r['expirationDate']) if price_df is not None else np.nan
    df['expiry_close'] = df.apply(expiry_close_from_cache, axis=1)

    # Compute labels and basic PnL (cash-settled approximation at expiry)
    def label_row(r):
        strike = safe_float(r["strike"])
        expiry_close = safe_float(r["expiry_close"])
        if not np.isfinite(strike) or not np.isfinite(expiry_close):
            return np.nan
        return 1 if expiry_close >= strike else 0  # win = OTM at expiry

    df["win"] = df.apply(label_row, axis=1)

    # Entry credit model: mid minus a fraction of half-spread
    def entry_credit(r, take_from_mid_pct=0.35, min_abs=0.01):
        bidPrice = safe_float(r["bidPrice"])
        #bid = safe_float(r["bid"]); ask = safe_float(r["ask"])
        #if not np.isfinite(bid) or not np.isfinite(ask) or bid<=0 or ask<=0:
        #    return np.nan
        
        #mid = 0.5*(bid+ask)
        mid = bidPrice
        #half_spread = max(0.0, (ask-bid)/2.0)
        half_spread = 0.0
        fill = mid - max(min_abs, take_from_mid_pct*half_spread)
        return max(0.0, fill)*100.0

    df["entry_credit"] = df.apply(entry_credit, axis=1)

    # Exit (expiry intrinsic) for puts
    def exit_intrinsic(r):
        strike = safe_float(r["strike"]) 
        expiry_close = safe_float(r["expiry_close"])
        if not np.isfinite(strike) or not np.isfinite(expiry_close):
            return np.nan
        return max(0.0, strike - expiry_close)*100.0

    df["exit_intrinsic"] = df.apply(exit_intrinsic, axis=1)

    # Capital reserved for CSP
    df["capital"] = derive_capital(df)

    # Total PnL and return
    df["total_pnl"] = df["entry_credit"] - df["exit_intrinsic"]
    df["return_pct"] = np.where(df["capital"]>0, df["total_pnl"]/df["capital"]*100.0, np.nan)

    return df

def main():

    data_dir = getenv("DATA_DIR", "")
    glob_pat = getenv("GLOB", "coveredPut_*.csv")
    target_time = getenv("TARGET_TIME", "11:00")
    batch_size = int(getenv("BATCH_SIZE", "30"))
    basic_csv = getenv("BASIC_CSV", "labeled_trades_normal.csv")

    cut_off_date = "2025-08-09"
    cut_off_date = pd.to_datetime(cut_off_date) if cut_off_date else None
    raw = load_csp_files(data_dir, glob_pat, target_time=target_time, enforce_daily_pick=True)
    # Preload price series with caching
    # out_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = getenv("CACHE_DIR", "./output")
    closes = preload_prices_with_cache(raw, cache_dir, batch_size=batch_size, cut_off_date=cut_off_date)
    labeled = build_dataset(raw, max_rows=0, preload_closes=closes)
    # Keep only rows that could be labeled (win not NaN)
    labeled = labeled[~labeled["win"].isna()].copy()

    out_dir = getenv("OUTPUT_DIR", "./output")
    labeled.to_csv(os.path.join(out_dir, basic_csv), index=False)

if __name__ == "__main__":
    main()
