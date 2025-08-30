#!/usr/bin/env python3
"""
a03merge_fundamentals_events.py  (.env version)

Join fundamentals (marketcap, sector) and earnings proximity features into a labeled trades file.

Configuration is read from .env file in the same directory (or project root).

.env example
------------
# === a03merge_fundamentals_events ===
TRADES_CSV=labeled_trades_normal.csv
EARNINGS_CSV=earnings_calendar.csv
FUNDAMENTALS_CSV=fundamentals.csv
OUTPUT_CSV=labeled_trades_enriched.csv

SYMBOL_COL=symbol        # optional, autodetect if blank
TRADE_DATE_COL=trade_date
SMALL_CAP=2000000000     # 2e9
LARGE_CAP=10000000000    # 1e10
DROP_DUPLICATES=true
"""

import os, sys, math
from typing import Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv


def load_config():
    load_dotenv()

    cfg = {
        "trades": os.getenv("TRADES_CSV"),
        "earnings": os.getenv("EARNINGS_CSV"),
        "fundamentals": os.getenv("FUNDAMENTALS_CSV"),
        "output": os.getenv("OUTPUT_CSV"),
        "symbol_col": os.getenv("SYMBOL_COL", None),
        "trade_date_col": os.getenv("TRADE_DATE_COL", "trade_date"),
        "small_cap": float(os.getenv("SMALL_CAP", 2e9)),
        "large_cap": float(os.getenv("LARGE_CAP", 1e10)),
        "drop_duplicates": os.getenv("DROP_DUPLICATES", "false").lower() in ("1","true","yes")
    }
    # Sanity check
    for k in ["trades","earnings","fundamentals","output"]:
        if not cfg[k]:
            raise ValueError(f"Missing required config {k.upper()} in .env")
    return cfg


def _detect_symbol_col(df: pd.DataFrame, user_symbol_col: str = None) -> str:
    if user_symbol_col:
        if user_symbol_col not in df.columns:
            raise KeyError(f"Symbol column '{user_symbol_col}' not found in trades CSV.")
        return user_symbol_col
    if "symbol" in df.columns:
        return "symbol"
    if "baseSymbol" in df.columns:
        return "baseSymbol"
    for alt in ["ticker", "underlying", "underlyingSymbol"]:
        if alt in df.columns:
            return alt
    raise KeyError("Could not auto-detect symbol column. Provide SYMBOL_COL in .env")


def _prep_earnings(df: pd.DataFrame, drop_dupes: bool) -> pd.DataFrame:
    sym_col = None
    for cand in ["symbol","ticker","baseSymbol"]:
        if cand in df.columns:
            sym_col = cand; break
    if not sym_col:
        raise KeyError("Earnings CSV must have 'symbol' or 'ticker' col")
    if "earnings_date" in df.columns:
        ed_col = "earnings_date"
    else:
        matches = [c for c in df.columns if c.lower().replace(" ","_") in ("earnings_date","earningsdate","date")]
        if matches: ed_col = matches[0]
        else: raise KeyError("Earnings CSV must have 'earnings_date' col")
    out = df.rename(columns={sym_col:"symbol", ed_col:"earnings_date"})[["symbol","earnings_date"]].copy()
    out["earnings_date"] = pd.to_datetime(out["earnings_date"], errors="coerce", utc=True).dt.tz_localize(None)
    out = out.dropna(subset=["symbol","earnings_date"])
    if drop_dupes: out = out.drop_duplicates(subset=["symbol","earnings_date"])
    return out.sort_values(["symbol","earnings_date"]).reset_index(drop=True)


def _prep_fundamentals(df: pd.DataFrame, drop_dupes: bool) -> pd.DataFrame:
    sym_col = None
    for cand in ["symbol","ticker","baseSymbol"]:
        if cand in df.columns: sym_col = cand; break
    if not sym_col: raise KeyError("Fundamentals CSV must have 'symbol' col")
    mc_col = None
    for cand in ["marketcap","marketCap","MarketCap"]:
        if cand in df.columns: mc_col = cand; break
    if not mc_col: raise KeyError("Fundamentals CSV must have 'marketcap' col")
    sector_col = None
    for cand in ["sector","Sector"]:
        if cand in df.columns: sector_col = cand; break
    out = df.rename(columns={sym_col:"symbol", mc_col:"marketcap"})[["symbol","marketcap"] + ([sector_col] if sector_col else [])].copy()
    out["marketcap"] = pd.to_numeric(out["marketcap"], errors="coerce")
    if sector_col: out = out.rename(columns={sector_col:"sector"})
    if drop_dupes:
        out = out.sort_values(["symbol"]).drop_duplicates(subset=["symbol"], keep="last")
    return out.reset_index(drop=True)


def _merge_next_prev_earnings(trades: pd.DataFrame, earnings: pd.DataFrame, symbol_col: str, trade_date_col: str):
    t = trades.copy()
    t[trade_date_col] = pd.to_datetime(t[trade_date_col], errors="coerce", utc=True).dt.tz_localize(None)
    if t[trade_date_col].isna().any():
        raise ValueError("Some trade_date values invalid")
    t = t.sort_values([symbol_col, trade_date_col]).reset_index(drop=True)
    e = earnings.copy().sort_values(["symbol","earnings_date"]).reset_index(drop=True)
    e.rename(columns={"symbol":symbol_col}, inplace=True)
    # next earnings
    # Next earnings (forward)
    next_e = pd.merge_asof(
        t,
        e.rename(columns={"earnings_date":"next_earnings_date"}),
        left_on=trade_date_col, right_on="next_earnings_date",
        by=symbol_col, direction="forward", allow_exact_matches=True
    )
    # Previous earnings (backward)
    prev_e = pd.merge_asof(
        t[[symbol_col,trade_date_col]].sort_values([symbol_col,trade_date_col]),
        e.rename(columns={"earnings_date":"prev_earnings_date"}),
        left_on=trade_date_col, right_on="prev_earnings_date",
        by=symbol_col, direction="backward", allow_exact_matches=True
    )
    combined = next_e.merge(prev_e[[symbol_col,trade_date_col,"prev_earnings_date"]],
                            on=[symbol_col,trade_date_col], how="left")
    return combined


def _add_feature_columns(df: pd.DataFrame, trade_date_col: str, small_cap: float, large_cap: float) -> pd.DataFrame:
    out = df.copy()
    for col in ["next_earnings_date","prev_earnings_date"]:
        if col in out.columns: out[col] = pd.to_datetime(out[col], errors="coerce")
    out["days_to_earnings"] = (out["next_earnings_date"] - out[trade_date_col]).dt.total_seconds()/(24*3600)
    out["is_earnings_week"] = out["days_to_earnings"].le(5).fillna(False)
    out["is_earnings_window"] = out["days_to_earnings"].le(10).fillna(False)
    days_since_prev = (out[trade_date_col]-out["prev_earnings_date"]).dt.total_seconds()/(24*3600)
    out["post_earnings_within_3d"] = days_since_prev.le(3).fillna(False)
    if "marketcap" in out.columns:
        out["log_marketcap"] = np.where(out["marketcap"]>0, np.log(out["marketcap"]), np.nan)
        def bucket(mcap):
            if pd.isna(mcap): return np.nan
            if mcap < small_cap: return "small"
            if mcap < large_cap: return "mid"
            return "large"
        out["cap_bucket"] = out["marketcap"].map(bucket)
    else:
        out["log_marketcap"], out["cap_bucket"] = np.nan, np.nan
    return out


def main():
    cfg = load_config()
    trades = pd.read_csv(cfg["trades"])
    symbol_col = _detect_symbol_col(trades, cfg["symbol_col"])
    earnings = _prep_earnings(pd.read_csv(cfg["earnings"]), cfg["drop_duplicates"])
    fundamentals = _prep_fundamentals(pd.read_csv(cfg["fundamentals"]), cfg["drop_duplicates"])
    merged = _merge_next_prev_earnings(trades, earnings, symbol_col, cfg["trade_date_col"])
    merged = merged.merge(fundamentals, on="symbol", how="left")
    enriched = _add_feature_columns(merged, cfg["trade_date_col"], cfg["small_cap"], cfg["large_cap"])
    new_cols = ["next_earnings_date","prev_earnings_date","days_to_earnings",
                "is_earnings_week","is_earnings_window","post_earnings_within_3d",
                "marketcap","log_marketcap","cap_bucket","sector"]
    ordered_cols = list(trades.columns)
    for c in new_cols:
        if c not in ordered_cols and c in enriched.columns: ordered_cols.append(c)
    for c in enriched.columns:
        if c not in ordered_cols: ordered_cols.append(c)
    enriched = enriched[ordered_cols]
    enriched.to_csv(cfg["output"], index=False)
    print(f"Wrote enriched dataset to: {cfg['output']}")
    print("Columns added:", [c for c in new_cols if c in enriched.columns])


if __name__ == "__main__":
    pd.set_option("display.width",140)
    pd.set_option("display.max_columns",200)
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
