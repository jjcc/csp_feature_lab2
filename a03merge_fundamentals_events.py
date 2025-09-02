#!/usr/bin/env python3
"""
a03merge_fundamentals_events.py  (HYBRID + Row‐preserving earnings merge; fundamentals removed)

This version **drops marketcap/sector entirely** ‐ no fundamentals merge, no
`log_marketcap` or `cap_bucket` engineering. Only earnings proximity features.

Precedence: CLI args > .env > defaults

Optional .env template:
-----------------------
TRADES_CSV=labeled_trades_normal.csv
EARNINGS_CSV=earnings_calendar.csv
OUTPUT_CSV=labeled_trades_enriched.csv

SYMBOL_COL=symbol
TRADE_DATE_COL=trade_date
DROP_DUPLICATES=true
STRICT_LENGTH_CHECK=true
"""

import argparse
import os, sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv


def _as_bool(x, default=False):
    if x is None: return default
    s = str(x).strip().lower()
    if s in ("1","true","yes","y","t"): return True
    if s in ("0","false","no","n","f"): return False
    return default


def _env_defaults():
    load_dotenv()
    return {
        "trades": os.getenv("TRADES_CSV", None),
        "earnings": os.getenv("EARNINGS_CSV", None),
        "output": os.getenv("OUTPUT_CSV", None),
        "symbol_col": os.getenv("SYMBOL_COL", None),
        "trade_date_col": os.getenv("TRADE_DATE_COL", "trade_date"),
        "drop_duplicates": _as_bool(os.getenv("DROP_DUPLICATES"), False),
        "strict_len_check": _as_bool(os.getenv("STRICT_LENGTH_CHECK"), True),
    }


def parse_args_with_env():
    env = _env_defaults()
    p = argparse.ArgumentParser(description="Enrich trades with leakage‐safe earnings proximity features. (CLI + .env hybrid)")
    p.add_argument("--trades", default=env["trades"], help="Trades CSV (default from .env TRADES_CSV)")
    p.add_argument("--earnings", default=env["earnings"], help="Earnings calendar CSV (default from .env EARNINGS_CSV)")
    p.add_argument("--output", default=env["output"], help="Output CSV (default from .env OUTPUT_CSV)")
    p.add_argument("--symbol-col", default=env["symbol_col"], help="Symbol column in trades (default from .env SYMBOL_COL or autodetect)")
    p.add_argument("--trade-date-col", default=env["trade_date_col"], help="Trade date column (default from .env TRADE_DATE_COL or 'trade_date')")
    # toggle-like behaviour for drop-duplicates based on env baseline
    if env["drop_duplicates"]:
        p.add_argument("--drop-duplicates", dest="drop_duplicates", action="store_false", help="Disable de-duplication (env default is ON)")
    else:
        p.add_argument("--drop-duplicates", dest="drop_duplicates", action="store_true", help="Enable de-duplication (env default is OFF)")
    if env["strict_len_check"]:
        p.add_argument("--no-strict-length-check", dest="strict_len_check", action="store_false",
                       help="Disable strict row-count check (env default is ON)")
    else:
        p.add_argument("--strict-length-check", dest="strict_len_check", action="store_true",
                       help="Enable strict row-count check (env default is OFF)")
    args = p.parse_args()

    # Validate required
    missing = [k for k in ["trades","earnings","output"] if not getattr(args, k)]
    if missing:
        raise SystemExit(f"Missing required path(s): {', '.join(missing)}. Provide via CLI or .env.")

    return args


def _detect_symbol_col(df: pd.DataFrame, user_symbol_col: str = None) -> str:
    if user_symbol_col:
        if user_symbol_col not in df.columns:
            raise KeyError(f"Symbol column '{user_symbol_col}' not found in trades CSV.")
        return user_symbol_col
    for cand in ["symbol", "baseSymbol", "ticker", "underlying", "underlyingSymbol"]:
        if cand in df.columns:
            return cand
    raise KeyError("Could not auto-detect symbol column. Provide --symbol-col or set SYMBOL_COL in .env")


def _prep_earnings(df: pd.DataFrame, drop_dupes: bool, symbol_col_for_merge: str) -> pd.DataFrame:
    # Standardize columns to ['symbol', 'earnings_date'] then rename 'symbol' -> symbol_col_for_merge
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
    
    # Ensure symbol is string type
    out["symbol"] = out["symbol"].astype(str)
    
    out["earnings_date"] = pd.to_datetime(out["earnings_date"], errors="coerce", utc=True).dt.tz_localize(None)
    out = out.dropna(subset=["symbol","earnings_date"])
    if drop_dupes: out = out.drop_duplicates(subset=["symbol","earnings_date"])
    out = out.sort_values(["symbol","earnings_date"]).reset_index(drop=True)
    # Rename 'symbol' to match trades' symbol column for merge_asof(by=...)
    out = out.rename(columns={"symbol": symbol_col_for_merge})
    return out



def _merge_next_prev_earnings(trades: pd.DataFrame, earnings: pd.DataFrame, symbol_col: str, trade_date_col: str, strict_len_check: bool):
    """Row-count‑preserving merge for next/prev earnings using merge_asof(by=...).

    Hardenings:
    - Exclude NaT/invalid rows from asof; carry them through with NaT matches.
    - Cast symbol columns to string (avoids mixed-type/grouping issues).
    - Sort LEFT by [symbol, trade_date] using mergesort (stable).
    - Sort RIGHT **after renaming** to the actual right_on names, also mergesort.
    - Fallback: if merge_asof still fails, do a per-symbol asof join.
    """
    t = trades.copy()
    n0 = len(t)
    t["_row_id"] = np.arange(n0)

    # Ensure symbol column is string
    if symbol_col in t.columns:
        t[symbol_col] = t[symbol_col].astype(str)

    # Parse trade dates; keep rows with NaT (they'll get NaT earnings)
    t[trade_date_col] = pd.to_datetime(t[trade_date_col], errors="coerce", utc=True).dt.tz_localize(None)

    bad_dates = t[trade_date_col].isna().sum()
    if bad_dates:
        print(f"[WARN] {bad_dates} trade rows have unparsable {trade_date_col}; carrying with NaT earnings.")

    # Valid subset for asof
    valid_mask = (~t[symbol_col].isna()) & (t[symbol_col] != "nan") & (~t[trade_date_col].isna())
    t_ok = t.loc[valid_mask].copy()
    t_na = t.loc[~valid_mask].copy()
    if len(t_na):
        print(f"[INFO] {len(t_na)} trade rows have NaN/invalid in '{symbol_col}' or '{trade_date_col}'; carried with NaT earnings.")

    if len(t_ok) == 0:
        combined = t.copy()
        combined["next_earnings_date"] = pd.NaT
        combined["prev_earnings_date"] = pd.NaT
        combined = combined.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"]).reset_index(drop=True)
        if strict_len_check and len(combined) != n0:
            raise AssertionError(f"[ASSERT] Row-count changed after merge: before={n0}, after={len(combined)}")
        print(f"[INFO] Earnings merge: rows before={n0}, after={len(combined)} (preserved, no valid rows).")
        return combined

    # RIGHT side: ensure types
    e = earnings.copy()
    e[symbol_col] = e[symbol_col].astype(str)

    # Sort LEFT by [symbol, trade_date]
    t_ok = t_ok.sort_values([symbol_col, trade_date_col], kind="mergesort").reset_index(drop=True)

    # Prepare RIGHT for NEXT: rename first, then sort on the new key
    e_next = e.rename(columns={"earnings_date": "next_earnings_date"})
    e_next = e_next.sort_values([symbol_col, "next_earnings_date"], kind="mergesort").reset_index(drop=True)

    # Helper: asof with per-symbol fallback
    def asof_with_fallback(left_df, right_df, right_on_name, direction):
        try:
            return pd.merge_asof(
                left_df,
                right_df,
                left_on=trade_date_col,
                right_on=right_on_name,
                by=symbol_col,
                direction=direction,
                allow_exact_matches=True
            )
        except ValueError as ex:
            print(f"[WARN] merge_asof({direction}) failed: {ex}. Falling back to per-symbol asof.")
            parts = []
            for sym, g in left_df.groupby(symbol_col, sort=False):
                r = right_df[right_df[symbol_col] == sym]
                if len(r) == 0:
                    g = g.copy()
                    g[right_on_name] = pd.NaT
                    parts.append(g)
                    continue
                g2 = g.sort_values(trade_date_col, kind="mergesort")
                r2 = r.sort_values(right_on_name, kind="mergesort")
                try:
                    joined = pd.merge_asof(
                        g2, r2,
                        left_on=trade_date_col, right_on=right_on_name,
                        direction=direction, allow_exact_matches=True
                    )
                except ValueError as inner_ex:
                    print(f"[WARN] per-symbol asof failed for {sym}: {inner_ex}. Filling NaT.")
                    g2 = g2.copy()
                    g2[right_on_name] = pd.NaT
                    joined = g2
                parts.append(joined)
            out = pd.concat(parts, ignore_index=True)
            return out.sort_values("_row_id", kind="mergesort")

    # NEXT earnings
    next_e = asof_with_fallback(t_ok, e_next, "next_earnings_date", "forward")

    # Prepare RIGHT for PREV
    e_prev = e.rename(columns={"earnings_date": "prev_earnings_date"})
    e_prev = e_prev.sort_values([symbol_col, "prev_earnings_date"], kind="mergesort").reset_index(drop=True)

    # Build base for prev merge (keep row_id)
    prev_base = next_e[[symbol_col, trade_date_col, "_row_id"]].sort_values([symbol_col, trade_date_col], kind="mergesort")
    prev_e = asof_with_fallback(prev_base, e_prev, "prev_earnings_date", "backward")

    # Combine
    combined_ok = next_e.merge(prev_e[["_row_id", "prev_earnings_date"]], on="_row_id", how="left", sort=False)

    # Add back invalid rows
    if len(t_na) > 0:
        for col in ["next_earnings_date", "prev_earnings_date"]:
            t_na[col] = pd.NaT
        combined = pd.concat([combined_ok, t_na], ignore_index=True)
    else:
        combined = combined_ok

    combined = combined.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"]).reset_index(drop=True)

    if strict_len_check and len(combined) != n0:
        raise AssertionError(f"[ASSERT] Row-count changed after merge: before={n0}, after={len(combined)}")

    print(f"[INFO] Earnings merge: rows before={n0}, after={len(combined)} (preserved).")
    return combined
def _add_feature_columns(df: pd.DataFrame, trade_date_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["next_earnings_date","prev_earnings_date"]:
        if col in out.columns: out[col] = pd.to_datetime(out[col], errors="coerce")
    out["days_to_earnings"] = (out["next_earnings_date"] - out[trade_date_col]).dt.total_seconds()/(24*3600)
    out["is_earnings_week"] = out["days_to_earnings"].le(5).fillna(False)
    out["is_earnings_window"] = out["days_to_earnings"].le(10).fillna(False)
    days_since_prev = (out[trade_date_col]-out["prev_earnings_date"]).dt.total_seconds()/(24*3600)
    out["post_earnings_within_3d"] = days_since_prev.le(3).fillna(False)
    return out


def main():
    args = parse_args_with_env()

    trades = pd.read_csv(args.trades)
    symbol_col = args.symbol_col or _detect_symbol_col(trades, None)
    earnings_raw = pd.read_csv(args.earnings)

    earnings = _prep_earnings(earnings_raw, args.drop_duplicates, symbol_col_for_merge=symbol_col)

    merged = _merge_next_prev_earnings(
        trades, earnings, symbol_col, args.trade_date_col, strict_len_check=args.strict_len_check
    )

    enriched = _add_feature_columns(merged, args.trade_date_col)

    # Column order: original + new
    new_cols = ["next_earnings_date","prev_earnings_date","days_to_earnings",
                "is_earnings_week","is_earnings_window","post_earnings_within_3d"]
    ordered_cols = list(trades.columns)
    for c in new_cols:
        if c not in ordered_cols and c in enriched.columns: ordered_cols.append(c)
    for c in enriched.columns:
        if c not in ordered_cols: ordered_cols.append(c)
    enriched = enriched[ordered_cols]

    if args.strict_len_check and len(enriched) != len(trades):
        raise AssertionError(f"[ASSERT] Final row-count changed: trades={len(trades)}, enriched={len(enriched)}")

    enriched[symbol_col] = merged[f"{symbol_col}_y"]
    enriched.drop(columns=[f"{symbol_col}_x", f"{symbol_col}_y"], inplace=True)

    enriched.to_csv(args.output, index=False)
    print(f"Wrote enriched dataset to: {args.output}")
    print("Columns added:", [c for c in new_cols if c in enriched.columns])


if __name__ == "__main__":
    pd.set_option("display.width",140)
    pd.set_option("display.max_columns",200)
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)