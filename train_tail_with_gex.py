#!/usr/bin/env python3
"""
Tail-Loss Classifier (v1, .env-driven)
--------------------------------------
This replicates the original "v1" trainer logic, but loads configuration
from a `.env` file instead of command-line arguments.

Key characteristics (same as v1):
- **Labeling:** tails are the *worst K% by dollar PnL* (no hybrid labels).
- **Model:** GradientBoostingClassifier (sklearn), fixed defaults.
- **Features:** BASE + GEX with median imputation (and special handling for `gex_missing`).
- **CV:** StratifiedKFold (default) or time-based splits for OOF predictions.
- **Artifacts:** compact joblib with model + medians + metadata, CSV of
  feature importances, and optional sampled OOF scores.

USAGE
-----
1) Create a `.env` (or copy the template provided alongside this script):
   cp tail_train_v1.example.env .env
   # then edit paths/params as needed

2) Run:
   python train_tail_with_gex_v1_env.py

ENV VARS
--------
CSV_INPUT           Path to input CSV (labeled_trades_with_gex.csv)
MODEL_OUT           Where to save model artifact (joblib)
IMP_OUT             Where to save feature importances CSV
SCORES_OUT          Where to save OOF scores CSV (sampled)
METRICS_OUT         Where to save metrics JSON
SAVE_SCORES_SAMPLE  Max rows to save in OOF scores CSV (0 to skip)
TAIL_PCT            Tail fraction by dollar PnL quantile (e.g., 0.03)
CV_TYPE             'stratified' (default) or 'time'
FOLDS               Number of folds/splits
SEED                Random seed for reproducibility
"""
import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from service.utils import prep_tail_training_df, ALL_FEATS

# Load .env (robustly try CWD and script dir)
def _load_env():
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    # Try current working dir first
    loaded = load_dotenv()
    if loaded:
        return
    # Then try alongside the script
    script_env = Path(__file__).resolve().parent / ".env"
    if script_env.exists():
        load_dotenv(dotenv_path=script_env)

_load_env()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

# --- Config ---
CSV_INPUT = os.getenv("CSV_INPUT", "/mnt/data/labeled_trades_with_gex.csv")
MODEL_OUT = os.getenv("MODEL_OUT", "/mnt/data/tail_model_gex_v1.pkl")
IMP_OUT = os.getenv("IMP_OUT", "/mnt/data/tail_gex_v1_feature_importances.csv")
SCORES_OUT = os.getenv("SCORES_OUT", "/mnt/data/tail_gex_v1_scores_oof.csv")
METRICS_OUT = os.getenv("METRICS_OUT", "/mnt/data/tail_train_metrics_v1.json")
SAVE_SCORES_SAMPLE = int(os.getenv("SAVE_SCORES_SAMPLE", "20000"))

TAIL_PCT = float(os.getenv("TAIL_PCT", "0.03"))
CV_TYPE = os.getenv("CV_TYPE", "stratified")
FOLDS = int(os.getenv("FOLDS", "5"))
SEED = int(os.getenv("SEED", "42"))



def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    # Parse times
    for c in ("tradeTime","expirationDate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # Derive total_pnl if missing (bid/bidPrice & expiry intrinsic)
    #if "total_pnl" not in df.columns:
    if True: # recalculate total_pnl anyway
        if "bid" not in df.columns and "bidPrice" in df.columns:
            df["bid"] = df["bidPrice"]
        bid_eff = df["bid"] if "bid" in df.columns else df["bidPrice"]
        df["entry_credit"] = pd.to_numeric(bid_eff, errors="coerce").fillna(0.0) * 100.0
        strike = pd.to_numeric(df["strike"], errors="coerce")
        expiry_close = pd.to_numeric(df["expiry_close"], errors="coerce")
        df["exit_intrinsic"] = np.maximum(strike - expiry_close, 0.0) * 100.0
        df["total_pnl"] = df["entry_credit"] - df["exit_intrinsic"]
        capital = pd.to_numeric(df["strike"], errors="coerce") * 100.0
        df["return_pct"] = 100.0 * df["total_pnl"] / capital
    return df

def _fill_features(df: pd.DataFrame, feat_list):
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
    return Xdf[ALL_FEATS].astype(float), medians

def main():
    # 1) Load & prep
    df = pd.read_csv(CSV_INPUT)
    #df = _prep_df(df).dropna(subset=["total_pnl"]).sort_values("tradeTime").reset_index(drop=True)
    df = prep_tail_training_df(df).dropna(subset=["total_pnl"]).sort_values("tradeTime").reset_index(drop=True)

    # 2) Features
    Xdf, medians = _fill_features(df, ALL_FEATS)
    X = Xdf.values

    # 3) Label tails (worst TAIL_PCT by dollar pnl)
    tail_cut = df["return_pct"].quantile(TAIL_PCT)
    y = (df["return_pct"] <= tail_cut).astype(int).values

    # Sanity check
    if y.sum() < max(5, FOLDS):
        raise SystemExit(f"Not enough tail examples ({y.sum()}) for folds={FOLDS}. Consider increasing TAIL_PCT.")

    # 4) OOF scoring
    if CV_TYPE == "time":
        splitter = TimeSeriesSplit(n_splits=FOLDS)
        splits = splitter.split(X)
    else:
        splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        splits = splitter.split(X, y)

    oof = np.zeros(len(df), dtype=float)
    importances = np.zeros(len(ALL_FEATS), dtype=float)
    n_models = 0

    for tr_idx, va_idx in splits:
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            continue

        clf = GradientBoostingClassifier(random_state=SEED)
        clf.fit(Xtr, ytr)
        oof[va_idx] = clf.predict_proba(Xva)[:, 1]
        importances += clf.feature_importances_
        n_models += 1

    if n_models == 0:
        raise SystemExit("All folds were degenerate (no positive labels). Increase TAIL_PCT or adjust CV_TYPE.")

    importances /= n_models

    # 5) Metrics
    auc = roc_auc_score(y, oof) if len(np.unique(y)) > 1 else float("nan")
    ap = average_precision_score(y, oof) if len(np.unique(y)) > 1 else float("nan")

    prec, rec, thr = precision_recall_curve(y, oof)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.nanargmax(f1))
    best_threshold = float(thr[best_idx - 1]) if 0 < best_idx < len(thr) else 0.5

    # 6) Final fit
    clf_all = GradientBoostingClassifier(random_state=SEED)
    clf_all.fit(X, y)

    # 7) Save artifacts
    for p in [MODEL_OUT, IMP_OUT, SCORES_OUT, METRICS_OUT]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": clf_all,
        "medians": medians,
        "features": ALL_FEATS,
        "tail_pct": float(TAIL_PCT),
        "tail_cut_return_pct": float(tail_cut), 
        "cv": CV_TYPE,
        "folds": int(FOLDS),
        "oof_auc": float(auc),
        "oof_avg_precision": float(ap),
        "oof_best_threshold": float(best_threshold),
    }, MODEL_OUT)

    (pd.DataFrame({"feature": ALL_FEATS, "importance": importances})
        .sort_values("importance", ascending=False)
        .to_csv(IMP_OUT, index=False))

    if SAVE_SCORES_SAMPLE > 0:
        oof_df = df[["baseSymbol","tradeTime","expirationDate","strike","total_pnl"]].copy()
        oof_df["tail_proba_oof"] = oof
        oof_df["is_tail"] = y
        if len(oof_df) > SAVE_SCORES_SAMPLE:
            oof_df = oof_df.sample(SAVE_SCORES_SAMPLE, random_state=SEED).sort_values("tradeTime")
        oof_df.to_csv(SCORES_OUT, index=False)

    with open(METRICS_OUT, "w") as f:
        json.dump({
            "rows": int(len(df)),
            "tails": int(int(y.sum())),
            "tail_pct": float(TAIL_PCT),
            "tail_cut": float(tail_cut),
            "cv": CV_TYPE,
            "folds": int(FOLDS),
            "oof_auc": float(auc),
            "oof_avg_precision": float(ap),
            "oof_best_threshold": float(best_threshold),
            "model_out": MODEL_OUT,
            "importances_out": IMP_OUT,
            "scores_out": SCORES_OUT if SAVE_SCORES_SAMPLE > 0 else None
        }, f, indent=2)

    print(f"[OK] Saved model → {MODEL_OUT}")
    print(f"     AUC={auc:.3f}, AP={ap:.3f}, best_threshold≈{best_threshold:.3f}")
    print(f"     Importances → {IMP_OUT}")
    if SAVE_SCORES_SAMPLE > 0:
        print(f"     OOF scores (sampled) → {SCORES_OUT}")
    print(f"     Metrics → {METRICS_OUT}")

if __name__ == "__main__":
    main()
