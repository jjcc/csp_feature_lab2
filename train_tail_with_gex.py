#!/usr/bin/env python3
"""
Train a tail-loss classifier (with optional GEX features) on labeled_trades_with_gex.csv.

- Labels "tail" trades as the worst K% by dollar PnL (per contract).
- Trains GradientBoostingClassifier with either time-based or stratified CV.
- Saves a compact model artifact with medians for scoring new data.
- (Optional) saves OOF scores (sampled) and feature importances.

USAGE (common):
  python train_tail_with_gex.py --csv labeled_trades_with_gex.csv --tail_pct 0.03 \
    --cv stratified --folds 5 --save_scores_sample 20000

USAGE (time-based CV):
  python train_tail_with_gex.py --csv labeled_trades_with_gex.csv --tail_pct 0.03 \
    --cv time --folds 3 --save_scores_sample 10000

Outputs (by default to current directory):
  - tail_model_gex.pkl                (model + medians + metadata)
  - tail_gex_feature_importances.csv  (feature importance)
  - tail_gex_scores_oof.csv           (OOF scores; sampled if requested)
  - tail_train_metrics.json           (AUC/AP/thresholds, etc.)
"""

import argparse, json, joblib, math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

BASE_FEATS = [
    "breakEvenProbability","moneyness","percentToBreakEvenBid","delta",
    "impliedVolatilityRank1y","potentialReturnAnnual","potentialReturn",
    "underlyingLastPrice","strike","openInterest","volume",
]
GEX_FEATS = [
    "gex_total","gex_total_abs","gex_pos","gex_neg",
    "gex_center_abs_strike","gex_flip_strike","gex_gamma_at_ul",
    "gex_distance_to_flip","gex_sign_at_ul","gex_missing",
]
ALL_FEATS = BASE_FEATS + GEX_FEATS

def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    # Parse times
    for c in ("tradeTime","expirationDate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # Robust PnL compute if missing
    if "total_pnl" not in df.columns:
        if "bid" not in df.columns and "bidPrice" in df.columns:
            df["bid"] = df["bidPrice"]
        # entry credit = bid * 100
        bid_eff = df["bid"] if "bid" in df.columns else df["bidPrice"]
        df["entry_credit"] = bid_eff.astype(float).fillna(0.0) * 100.0
        df["exit_intrinsic"] = np.maximum(df["strike"].astype(float) - df["expiry_close"].astype(float), 0.0) * 100.0
        df["total_pnl"] = df["entry_credit"] - df["exit_intrinsic"]
    return df

def _fill_features(df: pd.DataFrame, feat_list):
    medians = {}
    Xdf = df.copy()
    for c in feat_list:
        if c not in Xdf.columns:
            Xdf[c] = np.nan
        if c == "gex_missing":
            # treat missing-flag as 1 when NaN, and use 0 as median for imputation downstream
            Xdf[c] = Xdf[c].fillna(1)
            medians[c] = 0.0
        else:
            med = Xdf[c].median(skipna=True)
            medians[c] = float(med) if pd.notna(med) else 0.0
            Xdf[c] = Xdf[c].fillna(medians[c])
    return Xdf, medians

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to labeled_trades_with_gex.csv (or base CSV if no GEX)")
    ap.add_argument("--tail_pct", type=float, default=0.03, help="Tail fraction by dollar loss (default 3%)")
    ap.add_argument("--cv", choices=["time","stratified"], default="stratified", help="CV scheme")
    ap.add_argument("--folds", type=int, default=5, help="Number of folds/splits")
    ap.add_argument("--model_out", default="tail_model_gex.pkl")
    ap.add_argument("--imp_out", default="tail_gex_feature_importances.csv")
    ap.add_argument("--scores_out", default="tail_gex_scores_oof.csv")
    ap.add_argument("--metrics_out", default="tail_train_metrics.json")
    ap.add_argument("--save_scores_sample", type=int, default=20000,
                    help="Max rows to save in OOF scores file (sampled). Set 0 to skip saving scores.")
    args = ap.parse_args()

    # 1) Load & prep
    df = pd.read_csv(args.csv)
    df = _prep_df(df).dropna(subset=["total_pnl"]).sort_values("tradeTime").reset_index(drop=True)

    # 2) Features
    Xdf, medians = _fill_features(df, ALL_FEATS)
    X = Xdf[ALL_FEATS].astype(float).values

    # 3) Label tails (worst tail_pct by dollar pnl)
    tail_cut = df["total_pnl"].quantile(args.tail_pct)
    df["is_tail"] = (df["total_pnl"] <= tail_cut).astype(int)
    y = df["is_tail"].astype(int).values

    # Sanity checks
    if df["is_tail"].sum() < max(5, args.folds):
        raise SystemExit(f"Not enough tail examples ({df['is_tail'].sum()}) for folds={args.folds}. "
                         f"Try increasing --tail_pct (e.g., 0.05).")

    # 4) OOF scoring via chosen CV
    if args.cv == "time":
        if args.folds < 2:
            raise SystemExit("--folds must be >= 2 for time-based CV")
        splitter = TimeSeriesSplit(n_splits=args.folds)
    else:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    oof = np.zeros(len(df), dtype=float)
    importances = np.zeros(len(ALL_FEATS), dtype=float)
    n_models = 0

    for tr_idx, va_idx in splitter.split(X, y if args.cv=="stratified" else None):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        # Guard against degenerate folds (no tails in train or val)
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            # Skip this fold; continue
            continue

        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(Xtr, ytr)
        oof[va_idx] = clf.predict_proba(Xva)[:, 1]
        importances += clf.feature_importances_
        n_models += 1

    if n_models == 0:
        raise SystemExit("All folds were degenerate (no positive labels). Increase --tail_pct or adjust CV.")

    importances /= n_models

    # 5) Metrics & operating point
    auc = roc_auc_score(y, oof) if len(np.unique(y)) > 1 else float("nan")
    ap = average_precision_score(y, oof) if len(np.unique(y)) > 1 else float("nan")

    prec, rec, thr = precision_recall_curve(y, oof)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.nanargmax(f1))
    # sklearn's precision_recall_curve returns thresholds of length n-1
    best_threshold = float(thr[best_idx - 1]) if 0 < best_idx < len(thr) else 0.5

    # 6) Final refit on all data
    clf_all = GradientBoostingClassifier(random_state=42)
    clf_all.fit(X, y)

    # 7) Save artifacts
    joblib.dump({
        "model": clf_all,
        "medians": medians,
        "features": ALL_FEATS,
        "tail_pct": float(args.tail_pct),
        "tail_cut": float(tail_cut),
        "cv": args.cv,
        "folds": int(args.folds),
        "oof_auc": float(auc),
        "oof_avg_precision": float(ap),
        "oof_best_threshold": float(best_threshold),
    }, args.model_out)

    pd.DataFrame({"feature": ALL_FEATS, "importance": importances}) \
      .sort_values("importance", ascending=False) \
      .to_csv(args.imp_out, index=False)

    if args.save_scores_sample > 0:
        # Save sampled OOF scores to keep the file manageable
        oof_df = df[["baseSymbol","tradeTime","expirationDate","strike","total_pnl"]].copy()
        oof_df["tail_proba_oof"] = oof
        oof_df["is_tail"] = y
        if len(oof_df) > args.save_scores_sample:
            oof_df = oof_df.sample(args.save_scores_sample, random_state=42).sort_values("tradeTime")
        oof_df.to_csv(args.scores_out, index=False)

    with open(args.metrics_out, "w") as f:
        json.dump({
            "rows": int(len(df)),
            "tails": int(df["is_tail"].sum()),
            "tail_pct": float(args.tail_pct),
            "tail_cut": float(tail_cut),
            "cv": args.cv,
            "folds": int(args.folds),
            "oof_auc": float(auc),
            "oof_avg_precision": float(ap),
            "oof_best_threshold": float(best_threshold),
            "model_out": args.model_out,
            "importances_out": args.imp_out,
            "scores_out": args.scores_out if args.save_scores_sample > 0 else None
        }, f, indent=2)

    print(f"[OK] Saved model → {args.model_out}")
    print(f"     AUC={auc:.3f}, AP={ap:.3f}, best_threshold≈{best_threshold:.3f}")
    print(f"     Importances → {args.imp_out}")
    if args.save_scores_sample > 0:
        print(f"     OOF scores (sampled) → {args.scores_out}")
    print(f"     Metrics → {args.metrics_out}")

if __name__ == "__main__":
    main()
