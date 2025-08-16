#!/usr/bin/env python3
"""
score_tail_with_gex_env.py — (refactored to use model_utils)
"""
import os, json, joblib, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from service.utils import (
    ensure_dir,
    prep_tail_training_df,
    fill_features_with_training_medians,
    load_env_default,
)

def main():
    load_env_default()

    CSV_IN  = os.getenv("TAIL_SCORE_INPUT", "./new_trades_with_gex.csv")
    MODEL_IN= os.getenv("TAIL_MODEL_IN", "./tail_model_gex_v1.pkl")
    CSV_OUT = os.getenv("TAIL_SCORE_OUT", "./scores_tail.csv")
    PROBA_COL = os.getenv("TAIL_KEEP_PROBA_COL", "tail_proba")
    PRED_COL  = os.getenv("TAIL_PRED_COL", "is_tail_pred")
    THRESHOLD = os.getenv("TAIL_THRESHOLD", "")

    pack = joblib.load(MODEL_IN)
    clf = pack["model"]
    feats = pack["features"]
    med  = pack["medians"]
    default_thr = float(pack.get("oof_best_threshold", 0.5))

    thr = default_thr if (THRESHOLD is None or THRESHOLD.strip()=="") else float(THRESHOLD)

    # 1) Load and derive columns EXACTLY as in training
    raw = pd.read_csv(CSV_IN)
    df = prep_tail_training_df(raw)

    # 2) Fill features using TRAINING medians (and gex_missing rule)
    X, medians = fill_features_with_training_medians(df, feats)

    proba = clf.predict_proba(X)[:,1]
    out = df.copy()
    out[PROBA_COL] = proba

    if thr is not None:
        out[PRED_COL] = (out[PROBA_COL] >= thr).astype(int)

    ensure_dir(CSV_OUT)
    out.to_csv(CSV_OUT, index=False)

    summary = {
        "rows": int(len(out)),
        "threshold": float(thr),
        "tails_predicted": int(out[PRED_COL].sum()) if PRED_COL in out.columns else None,
        "kept_fraction_if_filter": float(1.0 - out[PRED_COL].mean()) if PRED_COL in out.columns else None
    }
    with open(Path(CSV_OUT).with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Scored {len(out)} rows. Saved → {CSV_OUT}")
    print(f"Threshold={thr:.6f}; predicted tails={summary['tails_predicted']}")

if __name__ == "__main__":
    main()
