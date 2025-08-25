#!/usr/bin/env python3
"""
score_winner_classifier_env.py

Score the input data using the winner classifier model.

Note:
Why I need score_winner_classifier_env.py?
This script  scores data using the winner classifier model separately. With another evaluation script I can compare the seperate scoring functions
behave the same way as the training so the training performance report is applicable to another separate scoring function in this script.
 *(refactored to use model_utils)
"""
import os, json, joblib, pandas as pd, numpy as np
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score
from service.utils import load_env_default, ensure_dir, prep_winner_like_training, pick_threshold_auto
from train_tail_with_gex import _add_dte_and_normalized_returns



def pick_threshold_from_coverage(proba, coverage):
    if len(proba) == 0: return 1.0
    k = max(1, int(round(len(proba) * coverage)))
    thr = np.partition(proba, len(proba)-k)[len(proba)-k]
    return float(thr)

def main():
    load_env_default()

    #CSV_IN  = os.getenv("WINNER_SCORE_INPUT", "./candidates.csv")
    #CSV_IN  = os.path.join(os.getenv("OUTPUT_DIR", "output"), os.getenv("MACRO_FEATURE_CSV", "./candidates.csv"))
    CSV_IN  =  os.getenv("MACRO_FEATURE_CSV", "./candidates.csv")
    #MODEL_IN= os.getenv("WINNER_MODEL_IN", "./output_winner/model_pack.pkl")
    MODEL_IN = os.getenv("WINNER_OUTPUT_DIR") + "/" + os.getenv("WINNER_MODEL_NAME")
    CSV_OUT = os.getenv("WINNER_SCORE_OUT", "./scores_winner.csv")
    PROBA_COL = os.getenv("WINNER_PROBA_COL", "prob_winner")
    PRED_COL  = os.getenv("WINNER_PRED_COL", "pred_winner")
    TRAIN_TARGET = os.getenv("WINNER_TRAIN_TARGET", "return_mon").strip()

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
    df = _add_dte_and_normalized_returns(df)
    if "tradeTime" in df.columns:
        df["tradeTime"] = pd.to_datetime(df["tradeTime"], errors="coerce")

    # Filter
    split_file = os.getenv("WINNER_SPLIT_FILE", "").strip()
    split_file = os.path.join(os.getenv("WINNER_OUTPUT_DIR", "output"), split_file)
    if split_file:
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"WINNER_SPLIT_FILE not found: {split_file}")
        df_split = pd.read_csv(split_file)
        if "tradeTime" in df_split.columns:
            df_split["tradeTime"] = pd.to_datetime(df_split["tradeTime"], errors="coerce")
        df = df.merge(df_split, on=["symbol", "tradeTime"], how="left") 
        # deal with the columns with "_x" in df
        col_x = [col for col in df.columns if col.endswith("_x")]
        for col in col_x:
            real_col = col[:-2]
            df[real_col] = df[col]
            df = df.drop(columns=[col, real_col+"_y"])
        # filter "is_train" == 0
        df = df[df["is_train"] == 0]


    # Match training-time preprocessing
    X, mask = prep_winner_like_training(df, feats, medians=medians, impute_missing=impute_missing)

    # Score
    proba = clf.predict_proba(X)[:,1]
    out = df.loc[mask].copy()
    out[PROBA_COL] = proba

    y = None
    if TRAIN_TARGET in out.columns:
        y = (pd.to_numeric(out[TRAIN_TARGET], errors="coerce") > 0).astype(int).values
    elif "win" in out.columns:
        y = out["win"].astype(int).values


    # Threshold selection
    if FIXED_THR.strip() != "":
        chosen_thr = float(FIXED_THR)
        thr_table = None
    elif AUTO_CAL and (TRAIN_TARGET in out.columns) and (targets_prec or targets_rec):
        y_true = (pd.to_numeric(out[TRAIN_TARGET], errors="coerce") > 0).astype(int).values
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
    OUT_SCORED = os.getenv("OUT_SCORED", (CSV_OUT or "data.csv").replace(".csv", "_scored.csv"))
    WRITE_SWEEP = os.getenv("WRITE_SWEEP", "1").strip().lower() in {"1","true","yes","y","on"}

    if WRITE_SWEEP:
        coverages = [0.10,0.20,0.30,0.40,0.50,0.60,0.70]
        rows = []
        for cov in coverages:
            thr = pick_threshold_from_coverage(proba, cov)
            mask = proba >= thr
            row = {"coverage": cov, "threshold": thr, "n": int(mask.sum())}
            if y is not None:
                row["precision_est"] = float(y[mask].mean()) if mask.any() else np.nan
                row["recall_est"] = float((y[mask]==1).sum()/max(1,(y==1).sum()))
            rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_SCORED.replace("_scored.csv", "_threshold_sweep.csv"), index=False)


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
    print(f"Threshold={chosen_thr:.6f}, coverage={summary['coverage']:.4f} for target precision {targets_prec} or recall {targets_rec}")

if __name__ == "__main__":
    main()
