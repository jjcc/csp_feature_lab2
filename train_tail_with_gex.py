#!/usr/bin/env python3
"""
Tail-Loss Classifier (v1, .env-driven)
--------------------------------------
This replicates the original "v1" trainer logic, but loads configuration
from a `.env` file instead of command-line arguments.
This extends the user's v1 trainer by optionally adding:
- VIX regime features
- 2-day & 5-day underlying cumulative returns (and normalized versions)

Key characteristics (same as v1):
- **Labeling:** tails are the *worst K% by dollar PnL* (no hybrid labels).
- **Model:** GradientBoostingClassifier (sklearn), fixed defaults.
- **Features:** BASE + GEX with median imputation (and special handling for `gex_missing`).
- **CV:** StratifiedKFold (default) or time-based splits for OOF predictions.
- **Artifacts:** compact joblib with model + medians + metadata, CSV of
  feature importances, and optional sampled OOF scores.
Key changes vs v1 (train_tail_with_gex.py):
- Robust feature engineering that avoids leakage by using daily closes up to trade date.
- Fix: _fill_features now returns Xdf[feat_list] (was using ALL_FEATS inadvertently).

(v1++ with VIX, Short-Term Moves, DTE & Normalized Returns)
Enhancements over v1_env_plus:
- Adds daysToExpiration (DTE) & log1p_DTE as features.
- Computes normalized returns:
    * return_per_day = return_pct / max(DTE, 1)
    * return_ann     = ((1 + return_pct/100)**(365/max(DTE,1)) - 1) * 100
- Lets you choose the labeling target via LABEL_ON: raw | per_day | annualized
  (default: per_day). The tail quantile cut is then applied on this chosen target.


USAGE
-----
1) Copy your .env or use the example below:
   # then edit paths/params as needed

CSV_INPUT=/mnt/data/labeled_trades_with_gex.csv
MODEL_OUT=/mnt/data/tail_model_gex_v1_plus.pkl
IMP_OUT=/mnt/data/tail_gex_v1_plus_feature_importances.csv
SCORES_OUT=/mnt/data/tail_gex_v1_plus_scores_oof.csv
METRICS_OUT=/mnt/data/tail_train_metrics_v1_plus.json
SAVE_SCORES_SAMPLE=20000
TAIL_PCT=0.03
CV_TYPE=stratified
FOLDS=5
SEED=42

2) Run:
   python train_tail_with_gex.py

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

Notes:
- Join features by calendar date (tradeTime.date()).
- Returns are cumulative sums of daily % returns over last 2/5 trading days.
- Normalized returns divide by 20-day rolling std of daily returns to stabilize scale.
"""
import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from service.utils import BASE_FEATS, GEX_FEATS, NEW_FEATS, prep_tail_training_df, ALL_FEATS
from service.preprocess import add_dte_and_normalized_returns

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
CSV_INPUT = os.getenv("OUTPUT_DIR") + "/" + os.getenv("MACRO_FEATURE_CSV", "/mnt/data/labeled_trades_with_gex.csv")
MODEL_OUT = os.getenv("MODEL_OUT", "/mnt/data/tail_model_gex_v1_plus.pkl")
IMP_OUT = os.getenv("IMP_OUT", "/mnt/data/tail_gex_v1_plus_feature_importances.csv")
SCORES_OUT = os.getenv("SCORES_OUT", "/mnt/data/tail_gex_v1_plus_scores_oof.csv")
METRICS_OUT = os.getenv("METRICS_OUT", "/mnt/data/tail_train_metrics_v1_plus.json")
SAVE_SCORES_SAMPLE = int(os.getenv("SAVE_SCORES_SAMPLE", "20000"))

TAIL_PCT = float(os.getenv("TAIL_PCT", "0.03"))
CV_TYPE = os.getenv("CV_TYPE", "stratified")
FOLDS = int(os.getenv("FOLDS", "5"))
SEED = int(os.getenv("SEED", "42"))
LABEL_ON = os.getenv("LABEL_ON", "per_day").lower()  # raw | per_day | annualized


DATE_COL = os.getenv("DATE_COL", "tradeTime")
PRICE_COL = os.getenv("PRICE_COL", "underlyingLastPrice")



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
    # FIX: return the columns requested by feat_list (not ALL_FEATS)
    return Xdf[feat_list].astype(float), medians

def main():
    # 1) Load & prep
    df = pd.read_csv(CSV_INPUT)
    #df = _prep_df(df).dropna(subset=["total_pnl"]).sort_values("tradeTime").reset_index(drop=True)
    # Use project utility for canonical prep (matches user's v1), then enrich.
    df = prep_tail_training_df(df).dropna(subset=["total_pnl"]).sort_values("tradeTime").reset_index(drop=True)
    #df = _build_macro_micro_features(df)
    df = add_dte_and_normalized_returns(df)

    # 2) Feature list: original + new (present only if computed)
    #{'VIX', 'ret_5d_norm', 'prev_close_minus_strike', 'ret_2d', 'prev_close_minus_strike_pct', 'log1p_DTE', 'prev_close', 'ret_5d', 'ret_2d_norm'}
    # NEW_FEATS = ["VIX", "ret_2d_norm", "ret_5d_norm",'prev_close_minus_strike_pct','log1p_DTE']
    #feat_list = list(BASE_FEATS + GEX_FEATS + NEW_FEATS)  # copy
    # use lean features after analysis
    LEAN_FEATS = [
        "potentialReturnAnnual", "VIX", "impliedVolatilityRank1y",
        "ret_2d_norm", "ret_5d_norm",
        "log1p_DTE", "daysToExpiration",
        "underlyingLastPrice",
        "gex_center_abs_strike", "gex_pos", "gex_gamma_at_ul",
        "potentialReturn",
        "prev_close_minus_ul_pct" 
    ]
    feat_list = LEAN_FEATS

    # 3) Build X and label tails by return_pct quantile
    Xdf, medians = _fill_features(df, feat_list)
    X = Xdf.values

    # 3) Label tails (worst TAIL_PCT by dollar pnl)
    if LABEL_ON == "raw":
        target_df = df["return_pct"]
    elif LABEL_ON == "annualized":
        target_df = df["return_ann"]
    elif LABEL_ON == "per_month":
        target_df = df["return_mon"]
    else: # default to per_day
        target_df = df["return_per_day"]

    #tail_cut = df["return_pct"].quantile(TAIL_PCT)
    tail_cut = target_df.quantile(TAIL_PCT)
    #y = (df["return_pct"] <= tail_cut).astype(int).values
    y = (target_df <= tail_cut).astype(int).values

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
    importances = np.zeros(len(feat_list), dtype=float)
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
        "features": feat_list,
        "tail_pct": float(TAIL_PCT),
        "label_on": LABEL_ON,
        "tail_cut_value": float(tail_cut),
        "cv": CV_TYPE,
        "folds": int(FOLDS),
        "oof_auc": float(auc),
        "oof_avg_precision": float(ap),
        "oof_best_threshold": float(best_threshold),
    }, MODEL_OUT)

    pd.DataFrame({"feature": feat_list, "importance": importances}) \
        .sort_values("importance", ascending=False) \
        .to_csv(IMP_OUT, index=False)

    if SAVE_SCORES_SAMPLE > 0:
        cols = ["baseSymbol","tradeTime","expirationDate","strike","total_pnl",
                "return_pct","return_per_day","return_ann","daysToExpiration"]
        have = [c for c in cols if c in df.columns]
        oof_df = df[have].copy()
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
            "label_on": LABEL_ON,
            "tail_cut_value": float(tail_cut),
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
    print(f"     Label target: {LABEL_ON}, tail_cut_value={tail_cut:.4f}")
    print(f"     Importances → {IMP_OUT}")
    if SAVE_SCORES_SAMPLE > 0:
        print(f"     OOF scores → {SCORES_OUT}")
    print(f"     Metrics → {METRICS_OUT}")

if __name__ == "__main__":
    main()
