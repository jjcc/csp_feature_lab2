#!/usr/bin/env python3
"""
Tail-Loss Classifier (v3, .env-driven)
--------------------------------------
This is the .env-config version of v2. It keeps the same modeling logic as v2 but
loads parameters from environment variables (via a .env file), not CLI flags.

Refinements vs v1:
- Flexible labeling: dollar-loss tail, percent-return tail, or hybrid (either condition).
- Time-based purged CV (embargo by days) to reduce temporal leakage.
- Cost-sensitive training via sample weights (heavier on tails; scale by loss magnitude).
- Recall-targeted threshold table (e.g., recall>=0.95/0.98) + best-F1.
- Optional lightweight GEX interactions for extra signal.
- Rolling (by month) OOF AUC/AP diagnostics to check stability.
- Saves a compact artifact (sklearn model + medians + metadata).

USAGE
-----
1) Create a ".env" file from the provided template:
   cp tail_train_v3.example.env .env
   # then edit paths and params as needed

2) Run:
   python train_tail_with_gex_v3_env.py

IMPORTANT: Expected input CSV should include (or allow derivation of):
- tradeTime, expirationDate (timestamps)
- strike, expiry_close (to derive intrinsic if total_pnl missing)
- bid or bidPrice (to derive entry credit if total_pnl missing)
- return_pct, capital (if missing, they will be derived when possible)
- GEX features (optional but recommended)

Outputs
-------
- MODEL_OUT: joblib artifact with {"model","medians","features",...}
- IMP_OUT: placeholder feature importances (HGBT has no built-in importances)
- SCORES_OUT: sampled OOF scores for inspection
- METRICS_OUT: JSON with OOF AUC/AP, best-F1 threshold, recall-target thresholds, rolling metrics
"""
import os, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

# =========================
# Config (.env parameters)
# =========================
def _pbool(x, default=False):
    return str(x).strip().lower() in {"1","true","y","yes","on"} if x is not None else default

CSV_INPUT = os.getenv("CSV_INPUT", "/mnt/data/labeled_trades_with_gex.csv")
MODEL_OUT = os.getenv("MODEL_OUT", "/mnt/data/tail_model_gex_v3.pkl")
IMP_OUT = os.getenv("IMP_OUT", "/mnt/data/tail_gex_v3_feature_importances.csv")
SCORES_OUT = os.getenv("SCORES_OUT", "/mnt/data/tail_gex_v3_scores_oof.csv")
METRICS_OUT = os.getenv("METRICS_OUT", "/mnt/data/tail_train_metrics_v3.json")

LABEL_BY = os.getenv("LABEL_BY", "hybrid")            # dollar|return|hybrid
TAIL_PCT_DOLLAR = float(os.getenv("TAIL_PCT_DOLLAR", "0.03"))
TAIL_PCT_RETURN = float(os.getenv("TAIL_PCT_RETURN", "0.03"))

CV_TYPE = os.getenv("CV_TYPE", "time_purged")         # time_purged|stratified
FOLDS = int(os.getenv("FOLDS", "4"))
EMBARGO_DAYS = int(os.getenv("EMBARGO_DAYS", "5"))

ADD_INTERACTIONS = _pbool(os.getenv("ADD_INTERACTIONS", "false"))

W_TAIL = float(os.getenv("W_TAIL", "8.0"))
ALPHA_RETURN_WEIGHT = float(os.getenv("ALPHA_RETURN_WEIGHT", "0.10"))

SEED = int(os.getenv("SEED", "42"))
SAVE_SCORES_SAMPLE = int(os.getenv("SAVE_SCORES_SAMPLE", "40000"))

# =============
# Feature sets
# =============
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

# =================
# Helper functions
# =================
def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    # Parse times
    for c in ("tradeTime","expirationDate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Derive total_pnl if missing
    if "total_pnl" not in df.columns:
        # choose entry credit proxy
        if "bid" not in df.columns and "bidPrice" in df.columns:
            df["bid"] = df["bidPrice"]
        bid_eff = df["bid"] if "bid" in df.columns else df.get("bidPrice", pd.Series(0, index=df.index))
        df["entry_credit"] = pd.to_numeric(bid_eff, errors="coerce").fillna(0.0) * 100.0
        strike = pd.to_numeric(df["strike"], errors="coerce")
        expiry_close = pd.to_numeric(df.get("expiry_close", np.nan), errors="coerce")
        df["exit_intrinsic"] = np.maximum(strike - expiry_close, 0.0) * 100.0
        df["total_pnl"] = df["entry_credit"] - df["exit_intrinsic"]

    # Capital & return_pct if missing
    if "capital" not in df.columns:
        df["capital"] = pd.to_numeric(df.get("strike", 0), errors="coerce") * 100.0
    if "return_pct" not in df.columns:
        cap = pd.to_numeric(df["capital"], errors="coerce")
        df["return_pct"] = np.where(cap>0, df["total_pnl"]/cap*100.0, np.nan)

    return df

def _fill_features(df: pd.DataFrame, feat_list, add_interactions: bool):
    medians = {}
    Xdf = df.copy()
    for c in feat_list:
        if c not in Xdf.columns:
            Xdf[c] = np.nan
        if c == "gex_missing":
            Xdf[c] = Xdf[c].fillna(1)
            medians[c] = 0.0
        else:
            med = Xdf[c].median(skipna=True)
            medians[c] = float(med) if pd.notna(med) else 0.0
            Xdf[c] = Xdf[c].fillna(medians[c])

    if add_interactions:
        if "gex_sign_at_ul" in Xdf and "moneyness" in Xdf:
            Xdf["gex_sign_moneyness"] = Xdf["gex_sign_at_ul"] * Xdf["moneyness"]
        if "gex_total" in Xdf and "delta" in Xdf:
            Xdf["gex_total_absdelta"] = Xdf["gex_total"] * Xdf["delta"].abs()
        if "gex_distance_to_flip" in Xdf and "moneyness" in Xdf:
            Xdf["gex_distflip_mny"] = Xdf["gex_distance_to_flip"] * Xdf["moneyness"]
    return Xdf, medians

def _label_tails(df: pd.DataFrame, label_by: str, tail_pct_dollar: float, tail_pct_return: float):
    dollar_cut = df["total_pnl"].quantile(tail_pct_dollar)
    is_tail_dollar = (df["total_pnl"] <= dollar_cut)
    ret_cut = df["return_pct"].quantile(tail_pct_return)
    is_tail_return = (df["return_pct"] <= ret_cut)
    if label_by == "dollar":
        is_tail = is_tail_dollar
    elif label_by == "return":
        is_tail = is_tail_return
    else:  # hybrid
        is_tail = is_tail_dollar | is_tail_return
    return is_tail.astype(int), float(dollar_cut), float(ret_cut)

def _time_purged_splits(times: pd.Series, n_splits: int, embargo_days: int):
    # Order by time, split into contiguous folds; embargo neighbors around validation window
    idx = np.arange(len(times))
    order = np.argsort(times.values)
    idx = idx[order]; t = times.values[order]
    folds = np.array_split(idx, n_splits)
    for i in range(n_splits):
        val_idx = folds[i]
        val_min = times.iloc[val_idx].min()
        val_max = times.iloc[val_idx].max()
        emb_before = val_min - pd.Timedelta(days=embargo_days)
        emb_after  = val_max + pd.Timedelta(days=embargo_days)
        train_mask = (times < emb_before) | (times > emb_after)
        train_idx = np.where(train_mask.values)[0]
        train_idx = np.setdiff1d(train_idx, val_idx)
        if len(train_idx)==0 or len(val_idx)==0:
            continue
        yield train_idx, val_idx

# =======
# Train
# =======
def main():
    # 1) Load & prep
    df = pd.read_csv(CSV_INPUT)
    df = _prep_df(df).dropna(subset=["total_pnl"]).sort_values("tradeTime").reset_index(drop=True)

    # 2) Labels
    y_series, dollar_cut, ret_cut = _label_tails(df, LABEL_BY, TAIL_PCT_DOLLAR, TAIL_PCT_RETURN)
    y = y_series.values

    # 3) Features (+ interactions)
    Xdf, medians = _fill_features(df, ALL_FEATS, ADD_INTERACTIONS)
    feats = list(Xdf.columns.intersection(ALL_FEATS + [
        "gex_sign_moneyness","gex_total_absdelta","gex_distflip_mny"
    ]))
    X = Xdf[feats].astype(float).values

    # 4) Weights (cost-sensitive)
    neg_ret = np.clip(pd.to_numeric(df["return_pct"], errors="coerce"), None, 0.0)
    w = np.where(y==1, W_TAIL + ALPHA_RETURN_WEIGHT * (-neg_ret), 1.0)
    w = np.clip(w, 1.0, 50.0)

    # 5) CV & OOF
    if CV_TYPE == "time_purged":
        splits = list(_time_purged_splits(df["tradeTime"], n_splits=FOLDS, embargo_days=EMBARGO_DAYS))
    else:
        splits = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(X, y))

    oof = np.full(len(df), np.nan, dtype=float)
    n_models = 0
    for tr_idx, va_idx in splits:
        if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[va_idx])) < 2:
            continue
        clf = HistGradientBoostingClassifier(random_state=SEED, max_depth=None)
        clf.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
        oof[va_idx] = clf.predict_proba(X[va_idx])[:,1]
        n_models += 1

    if n_models == 0:
        raise SystemExit("No valid folds after splitting; adjust tail pct or folds.")

    valid_mask = ~np.isnan(oof)
    auc = roc_auc_score(y[valid_mask], oof[valid_mask]) if len(np.unique(y[valid_mask])) > 1 else float("nan")
    ap = average_precision_score(y[valid_mask], oof[valid_mask]) if len(np.unique(y[valid_mask])) > 1 else float("nan")

    # 6) Threshold table
    prec, rec, thr = precision_recall_curve(y[valid_mask], oof[valid_mask])
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.nanargmax(f1)) if len(f1) else 0
    best_f1_threshold = float(thr[best_idx-1]) if 0 < best_idx < len(thr) else 0.5

    def threshold_for_recall(target):
        ok = np.where(rec >= target)[0]
        if len(ok)==0: return None
        i = ok[-1]
        return float(thr[i-1]) if 0 < i < len(thr) else (float(thr[0]) if len(thr)>0 else None)

    thr_90 = threshold_for_recall(0.90)
    thr_95 = threshold_for_recall(0.95)
    thr_98 = threshold_for_recall(0.98)

    # 7) Rolling stability by month
    oof_df = pd.DataFrame({
        "tradeTime": df["tradeTime"],
        "is_tail": y,
        "tail_proba_oof": oof
    }).dropna()
    oof_df["ym"] = oof_df["tradeTime"].dt.to_period("M")
    roll = []
    for ym, g in oof_df.groupby("ym"):
        if g["is_tail"].nunique() < 2:
            continue
        try:
            auc_m = roc_auc_score(g["is_tail"], g["tail_proba_oof"])
            ap_m = average_precision_score(g["is_tail"], g["tail_proba_oof"])
        except Exception:
            auc_m, ap_m = float("nan"), float("nan")
        roll.append({"ym": str(ym), "auc": auc_m, "ap": ap_m, "n": int(len(g))})
    roll_df = pd.DataFrame(roll)

    # 8) Final fit on full data
    final_clf = HistGradientBoostingClassifier(random_state=SEED, max_depth=None)
    final_clf.fit(X, y, sample_weight=w)

    # 9) Save artifacts
    Path(MODEL_OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(SCORES_OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(IMP_OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(METRICS_OUT).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": final_clf,
        "medians": {k: float(v) for k,v in medians.items()},
        "features": feats,
        "label_by": LABEL_BY,
        "tail_pct_dollar": float(TAIL_PCT_DOLLAR),
        "tail_pct_return": float(TAIL_PCT_RETURN),
        "dollar_cut": float(dollar_cut),
        "ret_cut": float(ret_cut),
        "cv": CV_TYPE,
        "folds": int(FOLDS),
        "embargo_days": int(EMBARGO_DAYS),
        "oof_auc": float(auc),
        "oof_avg_precision": float(ap),
        "oof_best_f1_threshold": float(best_f1_threshold),
        "thr_recall_90": thr_90,
        "thr_recall_95": thr_95,
        "thr_recall_98": thr_98,
        "w_tail": float(W_TAIL),
        "alpha_return_weight": float(ALPHA_RETURN_WEIGHT),
        "add_interactions": bool(ADD_INTERACTIONS)
    }, MODEL_OUT)

    # HGBT has no built-in importances; write placeholder to keep interface uniform
    pd.DataFrame({"feature": feats, "importance": np.nan}).to_csv(IMP_OUT, index=False)

    if SAVE_SCORES_SAMPLE != 0:
        out_scores = df[["baseSymbol","tradeTime","expirationDate","strike","total_pnl","return_pct"]].copy()
        out_scores["is_tail"] = y
        out_scores["tail_proba_oof"] = oof
        if SAVE_SCORES_SAMPLE > 0 and len(out_scores) > SAVE_SCORES_SAMPLE:
            out_scores = out_scores.sample(SAVE_SCORES_SAMPLE, random_state=SEED).sort_values("tradeTime")
        out_scores.to_csv(SCORES_OUT, index=False)

    metrics = {
        "rows": int(len(df)),
        "tails": int(int(y.sum())),
        "label_by": LABEL_BY,
        "tail_pct_dollar": float(TAIL_PCT_DOLLAR),
        "tail_pct_return": float(TAIL_PCT_RETURN),
        "dollar_cut": float(dollar_cut), "ret_cut": float(ret_cut),
        "cv": CV_TYPE, "folds": int(FOLDS), "embargo_days": int(EMBARGO_DAYS),
        "oof_auc": float(auc), "oof_avg_precision": float(ap),
        "best_f1_threshold": float(best_f1_threshold),
        "thresholds_for_recall": {"0.90": thr_90, "0.95": thr_95, "0.98": thr_98},
        "rolling_oof": roll_df.to_dict(orient="records")
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Saved model -> {MODEL_OUT}")
    print(f"     OOF AUC={auc:.3f}, AP={ap:.3f}, bestF1~{best_f1_threshold:.3f}")
    print(f"     Recall thresholds: 90%~{thr_90}, 95%~{thr_95}, 98%~{thr_98}")
    print(f"     Scores -> {SCORES_OUT}  Metrics -> {METRICS_OUT}")

if __name__ == "__main__":
    main()