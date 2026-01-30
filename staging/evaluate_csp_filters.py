
#!/usr/bin/env python3
"""
CSP Filter Evaluator
--------------------
Evaluate tail-risk reduction filters on your labeled CSP dataset.

Inputs:
  - labeled_trades.csv (default in current dir) OR pass --csv path

Outputs (written next to the script by default):
  - baseline_stats.json
  - single_feature_sweeps.csv
  - compound_filters.csv

Usage:
  python evaluate_csp_filters.py --csv /path/to/labeled_trades.csv
  # optional knobs:
  --min_coverage 0.2    # keep filters that retain >=20% of trades for single-feature sweeps
  --compound_min_coverage 0.15
"""

import argparse
import json
import numpy as np
import pandas as pd

def ensure_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Recompute returns if missing
    if "entry_credit" not in out.columns or out["entry_credit"].isna().all():
        bid = pd.to_numeric(out.get("bid"), errors="coerce")
        bidPrice = pd.to_numeric(out.get("bidPrice"), errors="coerce")
        out["entry_credit"] = np.where(~bid.isna(), bid * 100.0, np.where(~bidPrice.isna(), bidPrice * 100.0, np.nan))

    if "expiry_close" not in out.columns:
        raise SystemExit("Missing 'expiry_close' in labeled_trades.csv")

    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["expiry_close"] = pd.to_numeric(out["expiry_close"], errors="coerce")
    out["capital"] = out["strike"] * 100.0
    out["exit_intrinsic"] = np.maximum(0, out["strike"] - out["expiry_close"]) * 100.0
    out["total_pnl"] = out["entry_credit"] - out["exit_intrinsic"]
    out["return_pct"] = np.where(out["capital"]>0, out["total_pnl"]/out["capital"], np.nan)
    return out

def baseline_stats(df: pd.DataFrame) -> dict:
    return {
        "trades": int(len(df)),
        "win_rate": float((df["total_pnl"]>0).mean()),
        "avg_return_pct": float(df["return_pct"].mean()),
        "median_return_pct": float(df["return_pct"].median()),
        "worst_return_pct": float(df["return_pct"].min())
    }

def single_feature_sweeps(df: pd.DataFrame, min_coverage=0.2) -> pd.DataFrame:
    N = len(df)
    out_rows = []

    # Coerce numeric features we use
    for col in ["moneyness","percentToBreakEvenBid","impliedVolatilityRank1y","delta","breakEvenProbability"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def eval_mask(mask, feature, threshold, direction):
        sub = df[mask]
        if len(sub)==0: return None
        return {
            "feature": feature,
            "direction": direction,
            "threshold": threshold,
            "trades": int(len(sub)),
            "coverage_pct": float(len(sub)/N),
            "win_rate": float((sub["total_pnl"]>0).mean()),
            "avg_return_pct": float(sub["return_pct"].mean()),
            "median_return_pct": float(sub["return_pct"].median()),
            "worst_return_pct": float(sub["return_pct"].min())
        }

    rows = []

    if "moneyness" in df.columns:
        for th in [-0.10, -0.08, -0.06, -0.05, -0.04, -0.03]:
            r = eval_mask(df["moneyness"] <= th, "moneyness", th, "<=");  rows.append(r)

    if "percentToBreakEvenBid" in df.columns:
        for th in [-12, -10, -8, -6, -5, -4]:
            r = eval_mask(df["percentToBreakEvenBid"] <= th, "percentToBreakEvenBid", th, "<="); rows.append(r)

    if "delta" in df.columns:
        for th in [0.35, 0.30, 0.25, 0.20]:
            r = eval_mask(df["delta"] <= th, "delta", th, "<="); rows.append(r)

    if "breakEvenProbability" in df.columns:
        for th in [0.70, 0.72, 0.75, 0.78, 0.80]:
            r = eval_mask(df["breakEvenProbability"] >= th, "breakEvenProbability", th, ">="); rows.append(r)

    if "impliedVolatilityRank1y" in df.columns:
        for th in [40, 50, 60, 70]:
            rows.append(eval_mask(df["impliedVolatilityRank1y"] <= th, "impliedVolatilityRank1y", th, "<="))
            rows.append(eval_mask(df["impliedVolatilityRank1y"] >= th, "impliedVolatilityRank1y", th, ">="))

    rows = [r for r in rows if r is not None]
    sweeps = pd.DataFrame(rows)

    # Keep those that retain enough coverage
    return sweeps[sweeps["coverage_pct"]>=min_coverage].sort_values(
        ["worst_return_pct","avg_return_pct"], ascending=[False, False]
    )

def compound_filter_grid(df: pd.DataFrame, min_coverage=0.15) -> pd.DataFrame:
    N = len(df)
    # Coerce numeric
    for col in ["moneyness","percentToBreakEvenBid","impliedVolatilityRank1y","delta","breakEvenProbability"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    results = []
    # Anchor on BEP (high) + cushion (PBE negative) + deeper OTM + low delta + cap IVR
    bep_cuts = [0.76, 0.78, 0.80]
    pbe_cuts = [-10, -8, -6]
    mony_cuts = [-0.06, -0.05, -0.04]
    delta_cuts = [0.30, 0.25]
    ivr_caps = [60, 50, 40]

    for bth in bep_cuts:
        for pth in pbe_cuts:
            for mth in mony_cuts:
                for dth in delta_cuts:
                    for ivcap in ivr_caps:
                        mask = (
                            (df["breakEvenProbability"] >= bth) &
                            (df["percentToBreakEvenBid"] <= pth) &
                            (df["moneyness"] <= mth) &
                            (df["delta"] <= dth) &
                            (df["impliedVolatilityRank1y"] <= ivcap)
                        )
                        sub = df[mask]
                        if len(sub) >= min_coverage * N:
                            results.append({
                                "label": f"BEP≥{bth}, PBE≤{pth}%, mony≤{mth}, δ≤{dth}, IVR≤{ivcap}",
                                "trades": int(len(sub)),
                                "coverage_pct": float(len(sub)/N),
                                "win_rate": float((sub['total_pnl']>0).mean()),
                                "avg_return_pct": float(sub['return_pct'].mean()),
                                "median_return_pct": float(sub['return_pct'].median()),
                                "worst_return_pct": float(sub['return_pct'].min())
                            })
    if not results:
        return pd.DataFrame(columns=["label","trades","coverage_pct","win_rate","avg_return_pct","median_return_pct","worst_return_pct"])

    out = pd.DataFrame(results).sort_values(["worst_return_pct","avg_return_pct"], ascending=[False, False])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="labeled_trades.csv", help="Path to labeled_trades.csv")
    ap.add_argument("--min_coverage", type=float, default=0.20, help="Min coverage for single-feature sweeps")
    ap.add_argument("--compound_min_coverage", type=float, default=0.15, help="Min coverage for compound filters")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = ensure_returns(df)

    base = baseline_stats(df)
    print("=== BASELINE ===")
    for k,v in base.items():
        print(f"{k}: {v}")
    with open("baseline_stats.json", "w") as f:
        json.dump(base, f, indent=2)

    sweeps = single_feature_sweeps(df, min_coverage=args.min_coverage)
    sweeps.to_csv("single_feature_sweeps.csv", index=False)
    print("\nTop single-feature filters (tail-risk first):")
    print(sweeps.head(10).to_string(index=False))

    combos = compound_filter_grid(df, min_coverage=args.compound_min_coverage)
    combos.to_csv("compound_filters.csv", index=False)
    print("\nTop compound filters (tail-risk first):")
    print(combos.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
