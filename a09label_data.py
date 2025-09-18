
import os
from os import getenv
import numpy as np
import pandas as pd
from service.data_prepare import derive_capital, lookup_close_on_or_before, preload_prices_with_cache
from service.preprocess import load_csp_files 

from dotenv import load_dotenv
load_dotenv()




def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


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

def label_csv_file( raw):
    cut_off_date = "2025-09-06"
    cut_off_date = pd.to_datetime(cut_off_date) if cut_off_date else None
    batch_size = int(getenv("BATCH_SIZE", "30"))
    #processed_csv = getenv("BASIC_CSV", "labeled_trades_normal.csv")
    labeled_csv = getenv("OUTPUT_CSV", "labeled_trades_t1.csv")
    # Preload price series with caching
    cache_dir = getenv("CACHE_DIR", "./output")
    closes = preload_prices_with_cache(raw, cache_dir, batch_size=batch_size, cut_off_date=cut_off_date)
    labeled = build_dataset(raw, max_rows=0, preload_closes=closes)
    # Keep only rows that could be labeled (win not NaN)
    labeled = labeled[~labeled["win"].isna()].copy()
    out_dir = getenv("OUTPUT_DIR", "./output")
    labeled.to_csv(os.path.join(out_dir, labeled_csv), index=False)


def main():
    out_dir = getenv("OUT_DIR", "output")
    input_csv = getenv("MACRO_FEATURE_CSV", "trades_with_gex_macro.csv")
    input_csv = f"{out_dir}/{input_csv}"
    df = pd.read_csv(input_csv, index_col="row_id")

    label_csv_file(df)
if __name__ == "__main__":
    main()
