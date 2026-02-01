"""
Label option trade dataset with win/loss based on expiry prices.
There are two modes:
 1) single dataset labeling: label a single prepared dataset CSV file
 2) merged dataset labeling: label all merged datasets in the data_merged folder

The cutoff dates for labeling are read from the common configs.
"""
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Project root for absolute paths
PROJECT_ROOT = Path(__file__).parent
MISSING_STOCKS_PATH = PROJECT_ROOT / "data" / "missing_stocks.json"
EXCLUDE_STOCKS_PATH = PROJECT_ROOT / "data" / "exclude_stocks.json"

try:
    import exchange_calendars as xcals
    nyse = xcals.get_calendar("XNYS")
except Exception:
    nyse = None  # fall back to business-day heuristic below

from service.data_prepare import derive_capital,preload_prices_with_cache

from service.env_config import get_derived_file, getenv





def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def resolve_last_trading_session(expiry_ts: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(expiry_ts).tz_localize(None).normalize()
    # Prefer exchange calendar if available
    if nyse is not None:
        if nyse.is_session(d):
            sess = nyse.date_to_session(d, direction="none")
        else:
            sess = nyse.date_to_session(d, direction="previous")
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
    col = "Adj Close" if not use_unadjusted and "Adj Close" in price_df.columns else "Close"
    return float(price_df[col].get(session_date.normalize(), np.nan))


def build_dataset(raw: pd.DataFrame, max_rows: int = 0, preload_closes: dict = None) -> pd.DataFrame:
    """
    Prepare labeled dataset for modeling.
    Assumes columns (case-sensitive):
      baseSymbol, expirationDate, strike,  delta, moneyness, impliedVolatilityRank1y,
      potentialReturn, potentialReturnAnnual, breakEvenProbability, percentToBreakEvenBid,
      openInterest, volume, tradeTime, underlyingLastPrice
    Missing optional columns are filled with NaN.
    """
    df = raw.copy()

    # Validate required columns exist
    required_cols = ["baseSymbol", "expirationDate", "strike", "tradeTime", "underlyingLastPrice"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}. Cannot proceed with labeling.")

    # Standardize expected columns (optional ones can be filled with NaN)
    optional_cols = [
        "delta", "moneyness", "impliedVolatilityRank1y", "potentialReturn",
        "potentialReturnAnnual", "breakEvenProbability", "percentToBreakEvenBid",
        "openInterest", "volume", "__source_file"
    ]
    for c in optional_cols:
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
        if price_df is not None and price_df.columns.nlevels > 1:
            # MultiIndex 
            price_df = price_df[sym] # Extract symbol columns, drop MultiIndex

        return get_close_on_session(price_df, session, use_unadjusted=True)

    df['expiry_close'] = df.apply(expiry_close_from_cache, axis=1)

    # Entry credit model: mid minus a fraction of half-spread
    def entry_credit(r, take_from_mid_pct=0.35, min_abs=0.01):
        bidPrice = safe_float(r.get("bidPrice"))
        #bid = safe_float(r["bid"]); ask = safe_float(r["ask"])
        #if not np.isfinite(bid) or not np.isfinite(ask) or bid<=0 or ask<=0:
        #    return np.nan
        if not np.isfinite(bidPrice):
            return np.nan
        
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

    # Bin within each trade date (cross-sectional)
    df["_trade_date"] = df["tradeTime"].dt.tz_localize(None).dt.normalize()
    
    def assign_bins(g: pd.DataFrame) -> pd.DataFrame:
        s = g["return_pct"]
        # handle small groups safely
        if s.notna().sum() < 20:
            g["y_bin"] = np.nan
            return g
    
        # percentiles inside the day
        q25, q50, q75 = s.quantile([0.25, 0.50, 0.75]).values
    
        def to_bin(x):
            if not np.isfinite(x): return np.nan
            if x <= q25: return 0
            if x <= q50: return 1
            if x <= q75: return 2
            return 3
    
        g["y_bin"] = s.apply(to_bin)
        return g
    
    df = df.groupby("_trade_date", group_keys=False).apply(assign_bins)


    # add final win label based on return_pct threshold
    def label_row(r, win_threshold=0.88):
        return_pct = safe_float(r["return_pct"])
        if not np.isfinite(return_pct):
            return np.nan
        return 1 if return_pct > win_threshold else 0

    df["won"] = df.apply(label_row, axis=1)

    return df

def label_csv_file(raw, output_csv, cut_off_date=None):
    raw["expirationDate"] = pd.to_datetime(raw["expirationDate"], errors="coerce")
    raw_copy = raw.copy()
    cut_off_date = pd.to_datetime(cut_off_date).normalize()
    batch_size = int(getenv("DATA_BATCH_SIZE", "30"))
    #processed_csv = getenv("BASIC_CSV", "labeled_trades_normal.csv")
    labeled_csv = output_csv

    # Filter out trades with future expiration dates before labeling
    before_count = len(raw_copy)

    # Only keep trades that have expired by the  today #cut-off date
    # today = np.datetime64(pd.Timestamp.now().normalize())
    raw_copy = raw_copy[
        raw_copy["expirationDate"].notna() &
        (raw_copy["expirationDate"] <= cut_off_date)
        #(raw_copy["expirationDate"] <= today)
    ].copy()
    after_count = len(raw_copy)

    if before_count != after_count:
        print(f"Filtered out with cutoff {before_count - after_count} trades with expiration dates after {cut_off_date}")
        print(f"Remaining trades to label: {after_count}")
    
    # filter out trades with daysToExpiration > 14
    raw_copy = raw_copy[raw_copy["daysToExpiration"] <= 14]
    after_count2 = len(raw_copy)
    if after_count != after_count2:
        print(f"Filtered out with daysToExpiration > 14 : {after_count - after_count2} trades")
        print(f"Remaining trades to label: {after_count2}")

    # Preload price series with caching
    cache_dir = getenv("COMMON_OUTPUT_DIR", "./output")
    
    # modified to use syms, tt, ed instead of raw_copy to calculate inside the function
    syms = raw_copy['baseSymbol'].dropna().astype(str).str.upper().unique().tolist()
    tt = pd.to_datetime(raw_copy.get('tradeTime', pd.NaT), errors="coerce")
    ed = pd.to_datetime(raw_copy.get('expirationDate', pd.NaT), errors="coerce")

    closes = preload_prices_with_cache(
        syms, tt, ed, cache_dir, batch_size=batch_size, cut_off_date=cut_off_date
    )
    labeled = build_dataset(raw_copy, max_rows=0, preload_closes=closes)
    # Keep only rows that could be labeled (win not NaN)
    labeled = labeled[~labeled["won"].isna()].copy()
    print({
        "label_coverage": float(len(labeled) / max(len(raw_copy), 1)),
        "win_rate": float(labeled["won"].mean())
    })
    out_dir = getenv("COMMON_OUTPUT_DIR", "./output")
    out_dir = os.path.join(out_dir, "data_labeled")
    labeled.to_csv(os.path.join(out_dir, labeled_csv), index=False)


def main(merge_mode=False):
    if merge_mode:
        lablel_merge_dataset()
    else:
        #label_single_dataset()
        label_multiple_single_dataset()


def label_single_dataset():
    basic_csv = getenv("COMMON_DATA_BASIC_CSV", "trades_raw_orig.csv")
    out_dir = getenv("COMMON_OUTPUT_DIR", "output")
    out_dir = os.path.join(out_dir, "data_prep")
    input_csv, output_csv =  get_derived_file(basic_csv)
    input_csv = f"{out_dir}/{input_csv}"
    # filter rows with missing GEX if specified. Default: keep all rows
    if getenv("GEX_FILTER", "0").strip() in {"1","true","yes","y","on"}:
        input_csv = input_csv.replace(".csv", "_gexonly.csv")
    filtered_input_csv = os.path.join(os.path.dirname(input_csv), f"filtered_{os.path.basename(input_csv)}")
    if os.path.exists(filtered_input_csv):
        input_csv = filtered_input_csv
    df = pd.read_csv(input_csv, index_col="row_id")

    cut_off_date = getenv("COMMON_CUTOFF_DATE", "2025-09-29")
    label_csv_file(df, output_csv, cut_off_date)

def label_multiple_single_dataset():

    input_dir = getenv("COMMON_OUTPUT_DIR", "output")
    input_dir = os.path.join(input_dir, "data_prep")
    out_dir = getenv("COMMON_OUTPUT_DIR", "output")
    out_dir = os.path.join(out_dir, "data_labeled")
    os.makedirs(out_dir, exist_ok=True)

    cutoff_dates_by_tag = get_cutoff_dates()
    files = [f for f in os.listdir(input_dir) if f.startswith("filtered_trades_with_gex") and f.endswith(".csv")]
    files.sort()
    for f in files:
        #if '1027' not in f:
        if 'orig' not in f:
            continue # skip for investigation
        fpath = os.path.join(input_dir, f)
        print(f"Processing file: {fpath}")
        df = pd.read_csv(fpath, index_col="row_id")
        # remove the known missing stocks
        with open(MISSING_STOCKS_PATH, "r") as fp:
            missing_stocks = json.load(fp)
        df = df[~df['baseSymbol'].isin(missing_stocks)].copy()


        # get the cutoff date from the config
        last_tag = get_tag(f, merged = False)
        cutoff_date = cutoff_dates_by_tag.get(last_tag, None)
        print(f"  Cutoff date for tag {last_tag}: {cutoff_date}")
        output_csv = f"labeled_{f}"
        label_csv_file(df, output_csv, cutoff_date)

def lablel_merge_dataset():

    
    # inputs
    input_dir = getenv("COMMON_OUTPUT_DIR", "output")
    input_dir = os.path.join(input_dir, "data_merged")
    out_dir = getenv("COMMON_OUTPUT_DIR", "output")
    out_dir = os.path.join(out_dir, "data_labeled")
    os.makedirs(out_dir, exist_ok=True)

    cutoff_dates_by_tag = get_cutoff_dates()

    # get the files in the input_dir
    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    files.sort()
    for f in files:
        fpath = os.path.join(input_dir, f)
        print(f"Processing file: {fpath}")
        df = pd.read_csv(fpath, index_col="row_id")

        # get the cutoff date from the config
        last_tag = get_tag(f, merged = True)
        cutoff_date = cutoff_dates_by_tag.get(last_tag, None)
        print(f"  Cutoff date for tag {last_tag}: {cutoff_date}")
        output_csv = f"labeled_{f}"
        label_csv_file(df, output_csv, cutoff_date)

    
def get_tag(f, merged=False):
    if not merged:
        tag_block = f.split("_")[-1].replace(".csv", "")
        last_tag = tag_block
    else: # case of merged files
        tag_block = f.split("_")[-1].replace(".csv", "")
        if tag_block == "orig":
            last_tag = "orig"
        else:
            last_tag = tag_block[-1]
    return last_tag
    
def get_cutoff_dates():
    """
    Get the cutoff dates from the common configs
    Returns a dictionary of tag to cutoff date
    """
    from service.env_config import config 
    common_configs = config.get_common_configs_raw()
    # get the cutoff date for each tag
    cutoff_dates_by_tag = {}
    for k, v in common_configs.items():
        basic_csv = v.get("data_basic_csv", "N/A")
        file_name = basic_csv.replace(".csv", "")
        file_name_seg = file_name.split("_")
        group_tag = file_name_seg[file_name_seg.index("raw") +1]
        cutoff_date = v.get("cutoff_date", None)
        cutoff_dates_by_tag[group_tag] = cutoff_date
    return cutoff_dates_by_tag

if __name__ == "__main__":
    #merge_mode = True
    merge_mode = False  
    main(merge_mode=merge_mode)
