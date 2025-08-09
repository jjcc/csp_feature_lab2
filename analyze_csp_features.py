
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
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from dotenv import load_dotenv;
load_dotenv()

def ensure_cache_dir(out_dir):
    pc = os.path.join(out_dir, "price_cache")
    os.makedirs(pc, exist_ok=True)
    return pc

def load_cached_close_series(cache_dir, symbol):
    p = os.path.join(cache_dir, f"{symbol}.parquet")
    if os.path.exists(p):
        try:
            s = pd.read_parquet(p)
            if isinstance(s, pd.DataFrame) and "Close" in s.columns:
                s = s["Close"]
            s.index = pd.to_datetime(s.index)
            return s
        except Exception as e:
            print(f"[WARN] cache read failed for {symbol}: {e}")
    return None

def save_cached_close_series(cache_dir, symbol, close_series: pd.Series):
    try:
        df = close_series.to_frame("Close")
        df.to_parquet(os.path.join(cache_dir, f"{symbol}.parquet"))
    except Exception as e:
        print(f"[WARN] cache write failed for {symbol}: {e}")

def download_closes_batched(symbols, start_dt, end_dt, batch_size=30, threads=True):
    """
    Download daily Close prices for many symbols in batches using yfinance.
    Returns dict: symbol -> pd.Series (Close)
    """
    import yfinance as yf
    import math
    symbols = sorted(set([s for s in symbols if isinstance(s, str) and len(s)>0]))
    out = {}
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            df = yf.download(batch, start=start_dt.date(), end=(end_dt + pd.Timedelta(days=1)).date(),
                             interval="1d", group_by="ticker", auto_adjust=False, threads=threads, progress=False)
            # yfinance returns multi-index columns when multiple tickers
            if isinstance(df.columns, pd.MultiIndex):
                # df columns like ('AAPL','Close'), ('MSFT','Close'), ...
                for sym in batch:
                    if (sym, 'Close') in df.columns:
                        s = df[(sym, 'Close')].dropna()
                        s.index = pd.to_datetime(s.index)
                        out[sym] = s
            else:
                # single ticker case
                s = df['Close'].dropna()
                s.index = pd.to_datetime(s.index)
                out[batch[0]] = s
        except Exception as e:
            print(f"[WARN] batch download failed for {batch}: {e}")
    return out

def preload_closes_with_cache(raw_df, out_dir, batch_size=30):
    """
    From the raw CSP rows, determine unique symbols and date window,
    load from cache if available, batch-download missing ones, and save to cache.
    Returns dict: symbol -> Close series
    """
    cache_dir = ensure_cache_dir(out_dir)
    # Determine symbols and window
    syms = raw_df['baseSymbol'].dropna().astype(str).str.upper().unique().tolist()
    # Window: from min(tradeTime, expiry)-5 days to max(expiry)+1 day
    tt = pd.to_datetime(raw_df.get('tradeTime', pd.NaT), errors="coerce")
    ed = pd.to_datetime(raw_df.get('expirationDate', pd.NaT), errors="coerce")
    start_dt = pd.to_datetime(min([d for d in pd.concat([tt, ed.dropna()]) if pd.notna(d)]) - pd.Timedelta(days=5))
    end_dt = pd.to_datetime(max([d for d in ed.dropna()]) + pd.Timedelta(days=1))
    # Load from cache or mark missing
    closes = {}
    missing = []
    for s in syms:
        cs = load_cached_close_series(cache_dir, s)
        if cs is not None and (cs.index.min() <= start_dt) and (cs.index.max() >= end_dt):
            closes[s] = cs
        else:
            missing.append(s)
    if missing:
        print(f"[INFO] Downloading {len(missing)} symbols in batches (size={batch_size})...")
        fetched = download_closes_batched(missing, start_dt, end_dt, batch_size=batch_size, threads=True)
        for s, ser in fetched.items():
            closes[s] = ser
            save_cached_close_series(cache_dir, s, ser)
    return closes

def lookup_close_on_or_before(series: pd.Series, target_dt: pd.Timestamp) -> float:
    if series is None or series.empty or pd.isna(target_dt):
        return np.nan
    sub = series[series.index<=pd.to_datetime(target_dt)]
    if len(sub)==0:
        return np.nan
    return float(sub.iloc[-1])


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def get_expiry_close(symbol: str, expiry_ts: pd.Timestamp) -> float:
    """
    Fetch the underlying close on expiry day; if none (weekend/holiday), use the last close before expiry.
    """
    try:
        start = (expiry_ts - pd.Timedelta(days=5)).date()
        end = (expiry_ts + pd.Timedelta(days=1)).date()
        df = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
        if df.empty:
            return np.nan
        # pick the last close on or before expiry date
        mask = df.index.date <= expiry_ts.date()
        sub = df.loc[mask]
        if sub.empty:
            return np.nan
        return float(sub["Close"].iloc[-1])
    except Exception:
        return np.nan


def parse_timestamp_from_filename(name: str):
    """
    Expecting coveredPut_YYYY-MM-DD_HH_MM.csv or similar.
    Returns (date, time) as datetime.date, datetime.time or (None,None) if not parsed.
    """
    import re, datetime as dt
    m = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})', name)
    if not m:
        return None, None
    d = dt.datetime.strptime(m.group(1), "%Y-%m-%d").date()
    t = dt.time(int(m.group(2)), int(m.group(3)))
    return d, t

def pick_daily_snapshot_files(data_dir: str, pattern: str, target_time_str: str = "11:00"):
    """
    Groups files by date and picks one file per date:
    - choose the file with time >= target_time that is closest to it
    - if none, fall back to the latest file before target_time
    Returns a sorted list of absolute paths.
    """
    import glob, os, datetime as dt
    target_h, target_m = map(int, target_time_str.split(":"))
    target = dt.time(target_h, target_m)

    #paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    paths = sorted([str(Path(p)) for p in Path(data_dir).glob(pattern)])
    by_date = {}
    for p in paths:
        name = os.path.basename(p)
        d, t = parse_timestamp_from_filename(name)
        if d is None or t is None:
            continue
        by_date.setdefault(d, []).append((t, p))

    chosen = []
    for d, lst in by_date.items():
        lst.sort()  # sort by time
        after = [x for x in lst if x[0] >= target]
        if after:
            # pick nearest after target
            chosen.append(min(after, key=lambda x: (dt.datetime.combine(d, x[0]) - dt.datetime.combine(d, target)).total_seconds())[1])
        else:
            # pick latest before target
            chosen.append(max(lst, key=lambda x: x[0])[1])
    chosen = [str(Path(p)) for p in chosen]
    return sorted(chosen)

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

def load_csp_files(data_dir: str, pattern: str, target_time="11:00", enforce_daily_pick=True) -> pd.DataFrame:
    if enforce_daily_pick:
        paths = pick_daily_snapshot_files(data_dir, pattern, target_time)
    else:
        import glob, os
        paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            base_file = Path(p).name
            df["__source_file"] = base_file
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
    if not frames:
        raise SystemExit(f"No files found for pattern {pattern} in {data_dir}")
    return pd.concat(frames, ignore_index=True)



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

    # Use preloaded closes to compute expiry_close
    def expiry_close_from_cache(r):
        if preload_closes is None:
            return np.nan
        sym = str(r['baseSymbol']).upper()
        ser = preload_closes.get(sym)
        return lookup_close_on_or_before(ser, r['expirationDate']) if ser is not None else np.nan
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
        bid = safe_float(r["bid"]); ask = safe_float(r["ask"])
        if not np.isfinite(bid) or not np.isfinite(ask) or bid<=0 or ask<=0:
            return np.nan
        mid = 0.5*(bid+ask)
        half_spread = max(0.0, (ask-bid)/2.0)
        fill = mid - max(min_abs, take_from_mid_pct*half_spread)
        return max(0.0, fill)*100.0

    df["entry_credit"] = df.apply(entry_credit, axis=1)

    # Exit (expiry intrinsic) for puts
    def exit_intrinsic(r):
        strike = safe_float(r["strike"]); expiry_close = safe_float(r["expiry_close"])
        if not np.isfinite(strike) or not np.isfinite(expiry_close):
            return np.nan
        return max(0.0, strike - expiry_close)*100.0

    df["exit_intrinsic"] = df.apply(exit_intrinsic, axis=1)

    # Capital reserved for CSP
    df["capital"] = df["strike"].apply(safe_float)*100.0

    # Total PnL and return
    df["total_pnl"] = df["entry_credit"] - df["exit_intrinsic"]
    df["return_pct"] = np.where(df["capital"]>0, df["total_pnl"]/df["capital"]*100.0, np.nan)

    return df

def run_model(labeled: pd.DataFrame, out_dir: str):
    # Feature columns (only those that exist and are numeric)
    candidate_cols = [
        "moneyness","percentToBreakEvenBid","impliedVolatilityRank1y","delta",
        "potentialReturn","potentialReturnAnnual","breakEvenProbability",
        "openInterest","volume","underlyingLastPrice","strike"
    ]
    feats = [c for c in candidate_cols if c in labeled.columns]
    X = labeled[feats].apply(pd.to_numeric, errors="coerce")
    y = labeled["win"]

    mask = (~X.isna().any(axis=1)) & (~y.isna())
    X = X[mask]
    y = y[mask]

    if len(X) < 50:
        print(f"[WARN] Only {len(X)} clean rows after filtering; results may be unstable.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if y.nunique()>1 else None)
    clf = RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    # Save outputs
    labeled.to_csv(os.path.join(out_dir, "labeled_trades.csv"), index=False)
    importances = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
    importances.to_csv(os.path.join(out_dir, "feature_importances.csv"), header=["importance"])
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    print("\nTop features:\n", importances.head(10))

def main():
    #ap = argparse.ArgumentParser()
    #ap.add_argument("--data_dir", type=str, required=True, help="Folder containing CSP CSVs")
    #ap.add_argument("--glob", type=str, default="coveredPut_*.csv", help="Glob for CSP files")
    #ap.add_argument("--target_time", type=str, default="11:00", help="Target snapshot time HH:MM")
    #ap.add_argument("--all_snapshots", action="store_true", help="Use all files instead of one-per-day pick")
    #ap.add_argument("--max_rows", type=int, default=0, help="Limit rows per file (0 = no limit)")
    #args = ap.parse_args()

    data_dir = getenv("DATA_DIR", "")
    glob_pat = getenv("GLOB", "coveredPut_*.csv")
    target_time = getenv("TARGET_TIME", "11:00")
    batch_size = int(getenv("BATCH_SIZE", "30"))

    raw = load_csp_files(data_dir, glob_pat, target_time=target_time, enforce_daily_pick=True)
    # Preload price series with caching
    # out_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = getenv("OUTPUT_DIR", "./output")
    closes = preload_closes_with_cache(raw, out_dir, batch_size=batch_size)
    labeled = build_dataset(raw, max_rows=0, preload_closes=closes)
    # Keep only rows that could be labeled (win not NaN)
    labeled = labeled[~labeled["win"].isna()].copy()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    run_model(labeled, out_dir)

if __name__ == "__main__":
    main()
