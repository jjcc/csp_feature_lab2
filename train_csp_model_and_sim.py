
#!/usr/bin/env python3
"""
Train ML model to select CSP trades, then simulate portfolio performance.

Steps:
1) Load labeled_trades.csv and compute PnL if missing
2) Build features (BEP, moneyness, PBE, delta, IVR, liquidity, etc.)
3) Time-based split (by tradeTime): train early window, test later window
4) Train LogisticRegression (baseline) and GradientBoostingClassifier
5) Evaluate ROC-AUC, precision@top-k, avg return in top buckets
6) Pick probability threshold (or target coverage) and select trades
7) Run the v2 simulator on selected trades; save outputs

Usage:
  pip install -r requirements_ml.txt
  python train_csp_model_and_sim.py \
    --csv labeled_trades.csv \
    --initial_cash 100000 \
    --per_trade_capital_pct 0.30 \
    --daily_risk_cap_pct 1.00 \
    --max_concurrent 10 \
    --commission_per_contract 0.0 \
    --target_coverage 0.20 \
    --model gradient_boosting \
    --save_prefix mlgb

Outputs:
  - ml_metrics.json
  - proba_hist.csv
  - threshold_table.csv
  - selected_trades.csv
  - equity_curve_<prefix>.csv
  - portfolio_trades_<prefix>.csv
  - portfolio_summary_<prefix>.json
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# --- Minimal sim (fixed capital-release) ---
def simulate_variant_fixed(df: pd.DataFrame,
                     initial_cash: float,
                     per_trade_capital_pct: float,
                     daily_risk_cap_pct: float,
                     max_concurrent: int,
                     commission_per_contract: float,
                     sort_by: str = None,
                     sort_desc: int = 1,
                     contracts_per_trade: int = 1,
                     use_compounding: bool = True):
    df = df.dropna(subset=["tradeTime","expirationDate","capital","entry_credit","exit_intrinsic"]).copy()
    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(["tradeTime", sort_by], ascending=[True, not bool(sort_desc)])
    else:
        df = df.sort_values(["tradeTime"])

    equity = initial_cash
    cash = initial_cash
    t = []
    open_positions = []
    trades_log = []

    entries = df[["tradeTime"]].dropna().rename(columns={"tradeTime":"ts"})
    exits = df[["expirationDate"]].dropna().rename(columns={"expirationDate":"ts"})
    timeline = pd.concat([entries, exits]).drop_duplicates().sort_values("ts")["ts"].tolist()
    by_entry = {ts: g.copy() for ts, g in df.groupby(df["tradeTime"])}

    for ts in timeline:
        # Close positions first
        still_open = []
        for pos in open_positions:
            if pos["expirationDate"] <= ts:
                pnl = pos["total_pnl"] * pos["contracts"]
                fees = commission_per_contract * 2 * pos["contracts"]
                cash += pos["capital"] * pos["contracts"] + pnl - fees
                trades_log.append({
                    "symbol": pos.get("baseSymbol"),
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
        equity = cash

        # Open new positions
        if ts in by_entry:
            day_trades = by_entry[ts].copy()
            equity_for_alloc = equity if use_compounding else initial_cash
            per_trade_capital = per_trade_capital_pct * equity_for_alloc
            daily_cap = daily_risk_cap_pct * equity_for_alloc if daily_risk_cap_pct is not None else float("inf")
            slots = max(0, max_concurrent - len(open_positions))
            reserved_today = 0.0
            if slots > 0:
                selected = []
                for _, row in day_trades.iterrows():
                    need_cap = float(row["capital"]) * contracts_per_trade
                    if need_cap <= per_trade_capital and need_cap <= cash and (reserved_today + need_cap) <= daily_cap and slots > 0:
                        selected.append(row)
                        cash -= need_cap
                        reserved_today += need_cap
                        slots -= 1
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
    return pd.DataFrame(t).sort_values("ts").drop_duplicates("ts"), pd.DataFrame(trades_log)

def ensure_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tradeTime"] = pd.to_datetime(out["tradeTime"], errors="coerce")
    out["expirationDate"] = pd.to_datetime(out["expirationDate"], errors="coerce")
    # Entry credit
    if "entry_credit" not in out.columns or out["entry_credit"].isna().all():
        bid = pd.to_numeric(out.get("bid"), errors="coerce")
        bidPrice = pd.to_numeric(out.get("bidPrice"), errors="coerce")
        out["entry_credit"] = np.where(~bid.isna(), bid * 100.0, np.where(~bidPrice.isna(), bidPrice * 100.0, np.nan))
    # Intrinsic/Capital/PnL
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["expiry_close"] = pd.to_numeric(out["expiry_close"], errors="coerce")
    out["capital"] = out["strike"] * 100.0
    out["exit_intrinsic"] = np.maximum(0, out["strike"] - out["expiry_close"]) * 100.0
    out["total_pnl"] = out["entry_credit"] - out["exit_intrinsic"]
    out["return_pct"] = np.where(out["capital"]>0, out["total_pnl"]/out["capital"], np.nan)
    return out

def build_features(df: pd.DataFrame):
    # Select and clean features
    feats = [
        "breakEvenProbability","moneyness","percentToBreakEvenBid","delta","impliedVolatilityRank1y",
        "potentialReturnAnnual","potentialReturn","underlyingLastPrice","strike","openInterest","volume"
    ]
    X = df[feats].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # NaN handling: simple median impute (or drop rows)
    X = X.fillna(X.median(numeric_only=True))
    return X, feats

def time_based_split(df: pd.DataFrame, test_size=0.3):
    df = df.sort_values("tradeTime").dropna(subset=["tradeTime"])
    n = len(df)
    cut = int(n * (1 - test_size))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return train, test

def eval_topk(y_true, proba, returns, ks=(0.1, 0.2, 0.3)):
    out = []
    order = np.argsort(-proba)
    y = np.array(y_true)[order]
    r = np.array(returns)[order]
    for k in ks:
        m = max(1, int(len(y)*k))
        out.append({
            "top_k": k,
            "count": int(m),
            "win_rate": float(y[:m].mean()),
            "avg_return_pct": float(np.nanmean(r[:m]))
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="labeled_trades.csv")
    ap.add_argument("--initial_cash", type=float, default=100000.0)
    ap.add_argument("--per_trade_capital_pct", type=float, default=0.30)
    ap.add_argument("--daily_risk_cap_pct", type=float, default=1.00)
    ap.add_argument("--max_concurrent", type=int, default=10)
    ap.add_argument("--commission_per_contract", type=float, default=0.0)
    ap.add_argument("--contracts_per_trade", type=int, default=1)
    ap.add_argument("--model", type=str, default="gradient_boosting", choices=["logistic","gradient_boosting"])
    ap.add_argument("--target_coverage", type=float, default=0.20, help="Select top X%% by model prob for simulation")
    ap.add_argument("--save_prefix", type=str, default="ml")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = ensure_returns(df)
    # Define label: win at expiry
    df["win"] = (df["total_pnl"] > 0).astype(int)

    # Drop rows without key fields
    df = df.dropna(subset=["tradeTime","expirationDate","win"])
    # Train-test split by time
    train_df, test_df = time_based_split(df, test_size=0.3)

    # Features
    X_train, feat_cols = build_features(train_df)
    X_test, _ = build_features(test_df)
    y_train = train_df["win"].values
    y_test = test_df["win"].values

    # Models
    if args.model == "logistic":
        model = LogisticRegression(max_iter=200, n_jobs=None)
    else:
        model = GradientBoostingClassifier(random_state=42)

    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:,1]

    # Metrics
    roc = roc_auc_score(y_test, proba_test)
    ap = average_precision_score(y_test, proba_test)
    topk = eval_topk(y_test, proba_test, test_df["return_pct"].values, ks=(0.1,0.2,0.3))

    metrics = {
        "roc_auc": float(roc),
        "avg_precision": float(ap),
        "topk": topk,
        "features": feat_cols
    }
    with open("ml_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Thresholding by target coverage
    k = args.target_coverage
    thr_idx = int(len(proba_test) * (1-k))
    thr = np.sort(proba_test)[thr_idx] if len(proba_test)>0 else 1.0

    # Save probability histogram & threshold info
    hist_df = pd.DataFrame({"proba": proba_test}).sort_values("proba", ascending=False)
    hist_df.to_csv("proba_hist.csv", index=False)

    # Threshold sweep table for reference
    qs = [0.05,0.10,0.15,0.20,0.25,0.30]
    rows = []
    for q in qs:
        idx = int(len(proba_test)*(1-q))
        tthr = np.sort(proba_test)[idx] if len(proba_test)>0 else 1.0
        mask = proba_test >= tthr
        rows.append({
            "coverage": q,
            "threshold": float(tthr),
            "n": int(mask.sum()),
            "win_rate": float(y_test[mask].mean()) if mask.sum()>0 else np.nan,
            "avg_return_pct": float(np.nanmean(test_df["return_pct"].values[mask])) if mask.sum()>0 else np.nan
        })
    pd.DataFrame(rows).to_csv("threshold_table.csv", index=False)

    # Select top-k by proba on the WHOLE dataset (train+test) for sim, to mimic live deployment you'd refit on full data
    full_X, _ = build_features(df)
    model.fit(full_X, df["win"].values)
    joblib.dump(model, "ml_model.pkl")
    full_proba = model.predict_proba(full_X)[:,1]
    full_thr_idx = int(len(full_proba)*(1-args.target_coverage))
    full_thr = np.sort(full_proba)[full_thr_idx] if len(full_proba)>0 else 1.0
    df["ml_selected"] = (full_proba >= full_thr)

    selected = df[df["ml_selected"]].copy()
    selected.to_csv("selected_trades.csv", index=False)

    # Simulate selected trades
    curve, trades = simulate_variant_fixed(
        selected,
        initial_cash=args.initial_cash,
        per_trade_capital_pct=args.per_trade_capital_pct,
        daily_risk_cap_pct=args.daily_risk_cap_pct,
        max_concurrent=args.max_concurrent,
        commission_per_contract=args.commission_per_contract,
        sort_by="breakEvenProbability",
        sort_desc=1,
        contracts_per_trade=args.contracts_per_trade,
        use_compounding=True
    )
    eq_path = f"equity_curve_{args.save_prefix}.csv"
    tr_path = f"portfolio_trades_{args.save_prefix}.csv"
    sm_path = f"portfolio_summary_{args.save_prefix}.json"

    curve.to_csv(eq_path, index=False)
    trades.to_csv(tr_path, index=False)

    summary = {
        "ending_equity": float(curve["equity"].iloc[-1]) if not curve.empty else float(args.initial_cash),
        "total_return_pct": float((curve["equity"].iloc[-1] - args.initial_cash)/args.initial_cash*100.0) if not curve.empty else 0.0,
        "num_trades": int(len(trades)),
        "win_rate": float((trades["total_pnl_after_fees"] > 0).mean()) if not trades.empty else None,
        "avg_trade_pnl": float(trades["total_pnl_after_fees"].mean()) if not trades.empty else None,
        "worst_trade_pnl": float(trades["total_pnl_after_fees"].min()) if not trades.empty else None,
        "coverage_target": float(args.target_coverage),
        "threshold_used": float(full_thr)
    }
    with open(sm_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== ML EVAL ===")
    print(json.dumps(metrics, indent=2))
    print("\n=== SIM SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: ml_metrics.json, proba_hist.csv, threshold_table.csv, selected_trades.csv, {eq_path}, {tr_path}, {sm_path}")

if __name__ == "__main__":
    main()
