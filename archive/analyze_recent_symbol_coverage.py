
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
"""
The code is for analyzing recent symbol coverage in trading data.
== Coverage Summary ===
 N  days_evaluated  avg_hit_rate  median_hit_rate  min_hit_rate  max_hit_rate  avg_misses  avg_today_size  avg_prev_union_size
 3              43      0.797711         0.807292      0.677249      0.895954   36.302326      177.488372           285.488372
 4              43      0.829285         0.838926      0.730159      0.907514   30.604651      177.488372           321.534884
 5              43      0.851714         0.859223      0.767196      0.908163   26.558140      177.488372           352.093023
 
"""

def build_perday_sets(df: pd.DataFrame, date_col: str, symbol_col: str) -> pd.Series:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col, symbol_col])
    d[symbol_col] = d[symbol_col].astype(str).str.upper()
    d["trade_date"] = d[date_col].dt.floor("D")
    perday = (
        d.groupby("trade_date")[symbol_col]
         .apply(lambda s: set(s.dropna().tolist()))
         .sort_index()
    )
    return perday

def coverage_by_day(perday: pd.Series, N: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = perday.index[(perday.index >= start) & (perday.index <= end)]
    rows = []
    for t in dates:
        # Union of previous N trading days (strictly before t)
        prev_idx = perday.index[perday.index < t]
        if len(prev_idx) == 0:
            Uprev = set()
        else:
            prev_window = prev_idx[-N:] if len(prev_idx) >= N else prev_idx
            Uprev = set().union(*(perday[dd] for dd in prev_window)) if len(prev_window) else set()

        Stoday = perday[t] if t in perday.index else set()
        n_today = len(Stoday)
        n_prev_union = len(Uprev)
        hit = len(Stoday & Uprev)
        miss = n_today - hit
        hit_rate = hit / n_today if n_today > 0 else np.nan

        rows.append({
            "date": t,
            "N": N,
            "n_today": n_today,
            "n_prev_union": n_prev_union,
            "hits": hit,
            "misses": miss,
            "hit_rate": hit_rate,
        })
    return pd.DataFrame(rows).sort_values("date")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with at least [tradeTime, baseSymbol]")
    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD) inclusive")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD) inclusive")
    ap.add_argument("--days", nargs="+", type=int, default=[3,4,5], help="List of N values (previous N days)")
    ap.add_argument("--date_col", default="tradeTime")
    ap.add_argument("--symbol_col", default="baseSymbol")
    ap.add_argument("--outdir", default="./output")
    #args = ap.parse_args()

    args_input = "output/labeled_trades_normal.csv"
    args_start = "2025-06-08"
    args_end = "2025-08-08"
    args_days = [3, 4, 5]
    args_date_col = "tradeTime"
    args_symbol_col = "baseSymbol"
    args_outdir = "./output/diag"

    os.makedirs(args_outdir, exist_ok=True)

    df = pd.read_csv(args_input)



    perday = build_perday_sets(df, args_date_col, args_symbol_col)

    start = pd.to_datetime(args_start)
    end = pd.to_datetime(args_end)

    all_frames = []
    summary_rows = []

    for N in args_days:
        day_df = coverage_by_day(perday, N, start, end)
        out_path = os.path.join(args_outdir, f"coverage_by_day_N{N}.csv")
        day_df.to_csv(out_path, index=False)

        # Aggregate summary
        m = day_df["hit_rate"].dropna()
        summary_rows.append({
            "N": N,
            "days_evaluated": int(m.shape[0]),
            "avg_hit_rate": float(m.mean()) if not m.empty else np.nan,
            "median_hit_rate": float(m.median()) if not m.empty else np.nan,
            "min_hit_rate": float(m.min()) if not m.empty else np.nan,
            "max_hit_rate": float(m.max()) if not m.empty else np.nan,
            "avg_misses": float(day_df["misses"].mean()) if not day_df.empty else np.nan,
            "avg_today_size": float(day_df["n_today"].mean()) if not day_df.empty else np.nan,
            "avg_prev_union_size": float(day_df["n_prev_union"].mean()) if not day_df.empty else np.nan,
        })

        all_frames.append(day_df.assign(N=N))

    summary = pd.DataFrame(summary_rows).sort_values("N")
    summary_path = os.path.join(args_outdir, "coverage_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Optional print snippet
    print("=== Coverage Summary ===")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
