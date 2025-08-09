
#!/usr/bin/env python3
"""
CSP Portfolio Simulator
-----------------------
Simulate portfolio equity growth from a labeled CSP dataset (e.g., labeled_trades.csv).

Events:
- Entry at tradeTime (capital reserved = strike*100)
- Exit at expirationDate (PnL realized = entry_credit - exit_intrinsic)
- Optional commission per contract

Capital policy:
- Fixed initial cash, can reinvest profits (compounding)
- Per-trade allocation cap (% of initial or current equity)
- Max concurrent positions
- Selection per day: FIFO or sort by a column (e.g., breakEvenProbability desc)

Usage:
  python simulate_csp_portfolio.py --csv labeled_trades.csv \
      --initial_cash 100000 \
      --per_trade_capital_pct 0.1 \
      --max_concurrent 5 \
      --commission_per_contract 0.65 \
      --sort_by breakEvenProbability \
      --sort_desc 1

Outputs:
  - equity_curve.csv
  - portfolio_trades.csv
  - portfolio_summary.json
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime

def to_num(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def ensure_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Parse datetimes
    out["tradeTime"] = pd.to_datetime(out["tradeTime"], errors="coerce")
    out["expirationDate"] = pd.to_datetime(out["expirationDate"], errors="coerce")

    # Premium
    if "entry_credit" not in out.columns or out["entry_credit"].isna().all():
        bid = pd.to_numeric(out.get("bid"), errors="coerce")
        bidPrice = pd.to_numeric(out.get("bidPrice"), errors="coerce")
        out["entry_credit"] = np.where(~bid.isna(), bid * 100.0, np.where(~bidPrice.isna(), bidPrice * 100.0, np.nan))

    # Expiry close must exist
    if "expiry_close" not in out.columns:
        raise SystemExit("Missing 'expiry_close' in CSV")

    # Cash-secured values
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["expiry_close"] = pd.to_numeric(out["expiry_close"], errors="coerce")
    out["capital"] = out["strike"] * 100.0
    out["exit_intrinsic"] = np.maximum(0, out["strike"] - out["expiry_close"]) * 100.0
    out["total_pnl"] = out["entry_credit"] - out["exit_intrinsic"]
    out["return_pct"] = np.where(out["capital"]>0, out["total_pnl"]/out["capital"], np.nan)
    return out

def simulate(df: pd.DataFrame,
             initial_cash: float = 100000.0,
             per_trade_capital_pct: float = 0.1,
             max_concurrent: int = 5,
             commission_per_contract: float = 0.65,
             sort_by: str = None,
             sort_desc: int = 1,
             contracts_per_trade: int = 1,
             use_compounding: bool = True):
    """
    Event-driven simulation.
    - Each "trade" is 1 contract by default (set contracts_per_trade to increase size).
    - Capital reserved = strike*100 * contracts_per_trade.
    - Commission charged at open + close (2 legs) per contract.
    """
    df = df.copy()
    df = df.dropna(subset=["tradeTime","expirationDate","capital","entry_credit","exit_intrinsic"])
    # Optional selection sort within each calendar day
    df["trade_date"] = df["tradeTime"].dt.date
    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(["trade_date", sort_by], ascending=[True, not bool(sort_desc)])
        # Keep only the top N per day respecting max_concurrent when entries collide at same time later
    else:
        df = df.sort_values(["tradeTime"])  # FIFO fallback

    equity = initial_cash
    cash = initial_cash
    t = []  # equity curve
    open_positions = []  # list of dicts
    trades_log = []

    # Build global event timeline: all tradeTimes and expirationDates
    entries = df[["tradeTime"]].dropna().rename(columns={"tradeTime":"ts"})
    exits = df[["expirationDate"]].dropna().rename(columns={"expirationDate":"ts"})
    timeline = pd.concat([entries, exits]).drop_duplicates().sort_values("ts")["ts"].tolist()

    # Index trades by entry date
    by_entry = df.groupby(df["tradeTime"]).apply(lambda g: g.copy()).to_dict()

    for ts in timeline:
        # First, close positions expiring now
        still_open = []
        for pos in open_positions:
            if pos["expirationDate"] <= ts:
                # Realize PnL
                pnl = pos["total_pnl"] * pos["contracts"]
                fees = commission_per_contract * 2 * pos["contracts"]  # open + close
                cash += pnl - fees
                equity = cash  # no mark-to-market between events; could be extended
                trades_log.append({
                    "symbol": pos["baseSymbol"],
                    "entry": pos["tradeTime"],
                    "exit": pos["expirationDate"],
                    "strike": pos["strike"],
                    "contracts": pos["contracts"],
                    "entry_credit": pos["entry_credit"] * pos["contracts"],
                    "exit_intrinsic": pos["exit_intrinsic"] * pos["contracts"],
                    "fees": fees,
                    "total_pnl_after_fees": pnl - fees,
                    "capital_reserved": pos["capital"] * pos["contracts"],
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # Then, open new positions at this timestamp (if any)
        if ts in by_entry:
            day_trades = by_entry[ts].copy()
            # If compounding, allocation is % of current equity, otherwise % of initial
            equity_for_alloc = equity if use_compounding else initial_cash
            per_trade_capital = per_trade_capital_pct * equity_for_alloc

            # Enforce max concurrent: we can only open up to remaining slots
            slots = max(0, max_concurrent - len(open_positions))
            if slots <= 0:
                t.append({"ts": ts, "equity": equity, "cash": cash, "open_positions": len(open_positions)})
                continue

            # Select trades that fit capital and slots
            selected = []
            for _, row in day_trades.iterrows():
                need_cap = float(row["capital"]) * contracts_per_trade
                if need_cap <= per_trade_capital and need_cap <= cash and slots > 0:
                    selected.append(row)
                    cash -= need_cap
                    slots -= 1
                if slots <= 0:
                    break

            for row in selected:
                open_positions.append({
                    "baseSymbol": row.get("baseSymbol"),
                    "tradeTime": row["tradeTime"],
                    "expirationDate": row["expirationDate"],
                    "strike": float(row["strike"]),
                    "contracts": int(contracts_per_trade),
                    "entry_credit": float(row["entry_credit"]),
                    "exit_intrinsic": float(row["exit_intrinsic"]),
                    "total_pnl": float(row["total_pnl"]),
                    "capital": float(row["capital"])
                })

        t.append({"ts": ts, "equity": equity, "cash": cash, "open_positions": len(open_positions)})

    equity_curve = pd.DataFrame(t).sort_values("ts").drop_duplicates("ts")
    trades_log_df = pd.DataFrame(trades_log)

    # Summary
    if not trades_log_df.empty:
        total_return = (equity_curve["equity"].iloc[-1] - initial_cash) / initial_cash
        avg_trade_pnl = trades_log_df["total_pnl_after_fees"].mean()
        win_rate = (trades_log_df["total_pnl_after_fees"] > 0).mean()
        worst_trade = trades_log_df["total_pnl_after_fees"].min()
        invested_days = (equity_curve["ts"].iloc[-1] - equity_curve["ts"].iloc[0]).days
        summary = {
            "initial_cash": initial_cash,
            "ending_equity": float(equity_curve["equity"].iloc[-1]),
            "total_return_pct": float(total_return * 100.0),
            "num_trades": int(len(trades_log_df)),
            "win_rate": float(win_rate),
            "avg_trade_pnl": float(avg_trade_pnl),
            "worst_trade_pnl": float(worst_trade),
            "max_concurrent": int(max_concurrent),
            "per_trade_capital_pct": float(per_trade_capital_pct),
            "commission_per_contract": float(commission_per_contract),
            "use_compounding": bool(use_compounding),
            "period_days": int(invested_days)
        }
    else:
        summary = {"message": "No trades executed under provided constraints."}

    return equity_curve, trades_log_df, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="labeled_trades.csv")
    ap.add_argument("--initial_cash", type=float, default=100000.0)
    ap.add_argument("--per_trade_capital_pct", type=float, default=0.1, help="Max capital per trade as % of equity")
    ap.add_argument("--max_concurrent", type=int, default=5)
    ap.add_argument("--commission_per_contract", type=float, default=0.65)
    ap.add_argument("--sort_by", type=str, default=None, help="Column to sort within a day (e.g., breakEvenProbability)")
    ap.add_argument("--sort_desc", type=int, default=1, help="1 = descending, 0 = ascending")
    ap.add_argument("--contracts_per_trade", type=int, default=1)
    ap.add_argument("--no_compounding", action="store_true", help="If set, allocation uses initial cash (no compounding)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = ensure_returns(df)

    curve, trades_log, summary = simulate(
        df,
        initial_cash=args.initial_cash,
        per_trade_capital_pct=args.per_trade_capital_pct,
        max_concurrent=args.max_concurrent,
        commission_per_contract=args.commission_per_contract,
        sort_by=args.sort_by,
        sort_desc=args.sort_desc,
        contracts_per_trade=args.contracts_per_trade,
        use_compounding=(not args.no_compounding)
    )

    curve.to_csv("equity_curve.csv", index=False)
    trades_log.to_csv("portfolio_trades.csv", index=False)
    with open("portfolio_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nSaved: equity_curve.csv, portfolio_trades.csv, portfolio_summary.json")

if __name__ == "__main__":
    main()
