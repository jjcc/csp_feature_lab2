#!/usr/bin/env python3
"""
score_winner_classifier_env.py — (refactored to use model_utils)
"""
import os, json, joblib, pandas as pd, numpy as np
from pathlib import Path
from service.utils import load_env_default, ensure_dir, prep_winner_like_training, pick_threshold_auto

def main():
    load_env_default()

    CSV_IN  = os.getenv("WINNER_SCORE_INPUT", "./candidates.csv")
    MODEL_IN= os.getenv("WINNER_MODEL_IN", "./output_winner/model_pack.pkl")
    CSV_OUT = os.getenv("WINNER_SCORE_OUT", "./scores_winner.csv")
    PROBA_COL = os.getenv("WINNER_PROBA_COL", "prob_winner")
    PRED_COL  = os.getenv("WINNER_PRED_COL", "pred_winner")

    FIXED_THR = os.getenv("WINNER_SCORE_THRESHOLD", "")
    USE_PACK_F1 = str(os.getenv("WINNER_SCORE_USE_PACK_BEST_F1", "1")).lower() in {"1","true","yes","y","on"}

    TARGET_PREC = os.getenv("WINNER_SCORE_TARGET_PRECISION", "").strip()
    TARGET_RECALL= os.getenv("WINNER_SCORE_TARGET_RECALL", "").strip()
    AUTO_CAL = str(os.getenv("WINNER_SCORE_AUTO_CALIBRATE", "0")).lower() in {"1","true","yes","y","on"}

    targets_prec = [float(x.strip()) for x in TARGET_PREC.split(",") if x.strip()] if TARGET_PREC else []
    targets_rec  = [float(x.strip()) for x in TARGET_RECALL.split(",") if x.strip()] if TARGET_RECALL else []

    pack = joblib.load(MODEL_IN)
    clf = pack["model"]
    feats = pack["features"]
    medians = pack.get("medians", None)
    impute_missing = bool(pack.get("impute_missing", bool(medians is not None)))
    best_f1_thr = float(pack.get("metrics", {}).get("best_f1_threshold", 0.5))

    df = pd.read_csv(CSV_IN)

    # Match training-time preprocessing
    X, mask = prep_winner_like_training(df, feats, medians=medians, impute_missing=impute_missing)

    # Score
    proba = clf.predict_proba(X)[:,1]
    out = df.loc[mask].copy()
    out[PROBA_COL] = proba

    # Threshold selection
    if FIXED_THR.strip() != "":
        chosen_thr = float(FIXED_THR)
        thr_table = None
    elif AUTO_CAL and ("return_pct" in out.columns) and (targets_prec or targets_rec):
        y_true = (pd.to_numeric(out["return_pct"], errors="coerce") > 0).astype(int).values
        chosen_thr, thr_table = pick_threshold_auto(y_true, proba, targets_prec, targets_rec)
    elif USE_PACK_F1:
        chosen_thr = best_f1_thr
        thr_table = None
    else:
        chosen_thr = 0.5
        thr_table = None

    out[PRED_COL] = (out[PROBA_COL] >= chosen_thr).astype(int)

    ensure_dir(CSV_OUT)
    out.to_csv(CSV_OUT, index=False)

    summary = {
        "rows_scored": int(len(out)),
        "threshold": float(chosen_thr),
        "predicted_winners": int(out[PRED_COL].sum()),
        "coverage": float(out[PRED_COL].mean()),
    }
    with open(Path(CSV_OUT).with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)

    if thr_table is not None:
        thr_csv = Path(CSV_OUT).with_name(Path(CSV_OUT).stem + "_threshold_table.csv")
        thr_table.to_csv(thr_csv, index=False)

    print(f"[OK] Scored {len(out)} rows. Saved → {CSV_OUT}")
    print(f"Threshold={chosen_thr:.6f}, coverage={summary['coverage']:.4f}")

if __name__ == "__main__":
    main()
