
import os
import numpy as np
import pandas as pd

try:
    import exchange_calendars as xcals
    nyse = xcals.get_calendar("XNYS")
except Exception:
    nyse = None  # fall back to business-day heuristic below

from service.data_prepare import derive_capital, lookup_close_on_or_before, preload_prices_with_cache

from service.env_config import getenv





def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def resolve_last_trading_session(expiry_ts: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(expiry_ts).tz_localize(None).normalize()
    # Prefer exchange calendar if available
    if nyse is not None:
        # If expiry is a session, keep it; otherwise go to previous session
        sess = nyse.session_date(d) if nyse.is_session(d) else nyse.previous_session(d)
        return pd.Timestamp(sess).normalize()

    # Fallback: map weekends to Friday; if not Friday, step back one business day
    while d.weekday() > 4:          # 5=Sat, 6=Sun
        d -= pd.tseries.offsets.BDay(1)
    if d.weekday() != 4:            # not Friday -> previous business day
        d -= pd.tseries.offsets.BDay(1)
    return d.normalize()

def get_close_on_session(price_df, session_date, use_unadjusted=True):
    if price_df is None or len(price_df)==0:
        return np.nan
    if "date" in price_df.columns:
        idx = pd.to_datetime(price_df["date"]).dt.normalize()
        price_df = price_df.assign(_idx=idx).set_index("_idx")
    col = "close_unadj" if use_unadjusted and "close_unadj" in price_df.columns else "close"
    return float(price_df[col].get(session_date.normalize(), np.nan))


def build_dataset(raw: pd.DataFrame, max_rows: int = 0, preload_closes: dict = None) -> pd.DataFrame:
    """
    Prepare labeled dataset for modeling.
    Assumes columns (case-sensitive): 
      baseSymbol, expirationDate, strike,  delta, moneyness, impliedVolatilityRank1y, 
      potentialReturn, potentialReturnAnnual, breakEvenProbability, percentToBreakEvenBid,
      openInterest, volume, tradeTime, underlyingLastPrice
    Missing columns are tolerated and filled with NaN.
    """
    df = raw.copy()
    # Standardize expected columns
    expected_cols = [
        "baseSymbol","expirationDate","strike","delta","moneyness",
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

        expiry_date = pd.to_datetime(r["expirationDate"], errors="coerce")
        if pd.isna(expiry_date) or expiry_date > pd.Timestamp.now():
            return np.nan
        session = resolve_last_trading_session(expiry_date)
        sym = str(r["baseSymbol"]).upper()
        price_df = preload_closes.get(sym)
        return get_close_on_session(price_df, session, use_unadjusted=True)

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

def label_csv_file(raw):
    #cut_off_date = "2025-08-08"
    #cut_off_date = "2025-09-06"
    #cut_off_date = "2025-09-11" # the 3rd folder in "unprocessed3"
    cut_off_date = "2025-09-29" # the 3rd folder in "unprocessed3"
    cut_off_date = pd.to_datetime(cut_off_date) if cut_off_date else pd.Timestamp.now()
    batch_size = int(getenv("DATA_BATCH_SIZE", "30"))
    #processed_csv = getenv("BASIC_CSV", "labeled_trades_normal.csv")
    labeled_csv = getenv("COMMON_OUTPUT_CSV", "labeled_trades_t1.csv")

    # Filter out trades with future expiration dates before labeling
    raw_copy = raw.copy()
    raw_copy["expirationDate"] = pd.to_datetime(raw_copy["expirationDate"], errors="coerce")
    before_count = len(raw_copy)

    # Only keep trades that have expired by the  today #cut-off date
    #today = pd.Timestamp.now().normalize()
    # convert today to datetime64[ns]
    #today = today.astype("datetime64[ns]")
    today = np.datetime64(pd.Timestamp.now().normalize())
    raw_copy = raw_copy[
        raw_copy["expirationDate"].notna() &
        #(raw_copy["expirationDate"] <= cut_off_date)
        (raw_copy["expirationDate"] <= today)
    ].copy()
    after_count = len(raw_copy)

    if before_count != after_count:
        print(f"Filtered out {before_count - after_count} trades with expiration dates after {cut_off_date}")
        print(f"Remaining trades to label: {after_count}")

    # Preload price series with caching
    cache_dir = getenv("COMMON_OUTPUT_DIR", "./output")
    
    # modified to use syms, tt, ed instead of raw_copy to calculate inside the function
    syms = raw_copy['baseSymbol'].dropna().astype(str).str.upper().unique().tolist()
    tt = pd.to_datetime(raw_copy.get('tradeTime', pd.NaT), errors="coerce")
    ed = pd.to_datetime(raw_copy.get('expirationDate', pd.NaT), errors="coerce")

    closes = preload_prices_with_cache(syms, tt, ed, cache_dir, batch_size=batch_size, cut_off_date=cut_off_date)
    labeled = build_dataset(raw_copy, max_rows=0, preload_closes=closes)
    # Keep only rows that could be labeled (win not NaN)
    labeled = labeled[~labeled["win"].isna()].copy()
    out_dir = getenv("COMMON_OUTPUT_DIR", "./output")
    labeled.to_csv(os.path.join(out_dir, labeled_csv), index=False)


def main():
    out_dir = getenv("COMMON_OUTPUT_DIR", "output")
    input_csv = getenv("COMMON_MACRO_FEATURE_CSV", "trades_with_gex_macro.csv")
    input_csv = f"{out_dir}/{input_csv}"
    # filter rows with missing GEX if specified. Default: keep all rows
    if getenv("GEX_FILTER", "0").strip() in {"1","true","yes","y","on"}:
        input_csv = input_csv.replace(".csv", "_gexonly.csv")
    df = pd.read_csv(input_csv, index_col="row_id")

    label_csv_file(df)
if __name__ == "__main__":
    main()
