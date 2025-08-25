#!/usr/bin/env python3
"""Score tail + winner pipeline as a standalone script.

This is a converted version of the test `test_score_tail_winner_classifier` from
`test/test_prod.py` made into a runnable script placed at the repository root.
"""
from glob import glob
import joblib
import pandas as pd
import os
from datetime import time

from dotenv import load_dotenv

from service.preprocess import merge_gex
from service.utils import fill_features_with_training_medians, prep_tail_training_df, prep_winner_like_training

load_dotenv()

TAIL_MODEL_IN = "models/tail_model_gex_v2_cut05.pkl"
TAIL_KEEP_PROBA_COL = "tail_proba"
WINNER_MODEL_IN = "output/winner/model_pack.pkl"
WINNER_PROBA_COL = "proba"


def parse_target_time(s: str) -> time:
    try:
        hh, mm = s.split(":")
        return time(int(hh), int(mm))
    except Exception:
        return time(11, 0)


def get_merge_params():
    option_data_dir = "option/put"
    glob_pat = f"coveredPut_*.csv"
    # get the latest file
    latest_file_with_path = max(glob(os.path.join(option_data_dir, glob_pat)), key=os.path.getctime)
    latest_file = os.path.basename(latest_file_with_path)
    process_log = "log/processed.log"
    # read processed log (create if missing)
    if not os.path.exists(process_log):
        open(process_log, "a").close()
    with open(process_log, "r") as f:
        lines = f.readlines()
    files = [line.strip() for line in lines if line.strip()]
    if latest_file in files:
        print(f"Latest file {latest_file} already processed.")
        return None, None
    # get time of latest_file, like coveredPut_2025-08-15_16_30.csv
    latest_file_time = latest_file.split("_")[-2:]
    latest_file_time = ":".join(latest_file_time).replace(".csv", "")
    target_t = parse_target_time(latest_file_time)
    target_minutes = target_t.hour * 60 + target_t.minute
    return latest_file_with_path, target_minutes


def main():
    latest_file_with_path, target_minutes = get_merge_params()
    if latest_file_with_path is None:
        print("File is processed or no file found.")
        return

    option_file = latest_file_with_path

    # construct an output file path
    hour, minute = option_file.replace(".csv", "").split("_")[-2:]
    target_date = option_file.split("_")[1]
    time_str = f"{hour}_{minute}"
    out_dir = "prod/output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/scored_tail_winner_test_{target_date}_{time_str}.csv"

    gex_base_dir = os.getenv("GEX_BASE_DIR")
    df_o = merge_gex(option_file, gex_base_dir, target_minutes=target_minutes)

    # tail scoring
    pack_tl = joblib.load(TAIL_MODEL_IN)
    clf_tl = pack_tl["model"]
    feats = pack_tl["features"]
    med = pack_tl.get("medians", None)
    df = prep_tail_training_df(df_o)
    X_tl, medians = fill_features_with_training_medians(df, feats)

    proba = clf_tl.predict_proba(X_tl)[:, 1]
    out = df.copy()
    out[TAIL_KEEP_PROBA_COL] = proba
    thresh = 0.04028
    out["is_tail_pred"] = (out[TAIL_KEEP_PROBA_COL] >= thresh).astype(int)

    # winner scoring
    pack_wc = joblib.load(WINNER_MODEL_IN)
    clf_wc = pack_wc["model"]
    feats = pack_wc["features"]
    medians = pack_wc.get("medians", None)
    impute_missing = bool(pack_wc.get("impute_missing", bool(medians is not None)))
    Xwc, mask = prep_winner_like_training(df, feats, medians=medians, impute_missing=impute_missing)

    proba = clf_wc.predict_proba(Xwc)[:, 1]
    out2 = out.loc[mask].copy()
    out2[WINNER_PROBA_COL] = proba
    thresh = 0.8375
    out2["is_winner_pred"] = (out2[WINNER_PROBA_COL] >= thresh).astype(int)

    # cleanup and filters
    gex_columns = [col for col in out2.columns if col.startswith("gex_")]
    out2.drop(columns=gex_columns, inplace=True, errors='ignore')
    to_drop = [
        'baseSymbolType', 'expirationDate',
        'strike', 'moneyness', 'breakEvenBid', 'percentToBreakEvenBid', 'tradeTime', 'symbol_norm', 'impliedVolatilityRank1y',
        'delta', 'breakEvenProbability', 'expirationType', 'symbolType',
        'entry_credit', 'exit_intrinsic', 'total_pnl', 'return_pct'
    ]
    out2.drop(columns=to_drop, inplace=True, errors='ignore')
    out2 = out2[out2["is_winner_pred"] == 1]
    out2 = out2[out2["is_tail_pred"] == 0]
    out2.to_csv(out_path, index=False)

    # log processing information
    process_log = "log/processed.log"
    latest_file = os.path.basename(latest_file_with_path)
    with open(process_log, "a") as f:
        f.write(f"{latest_file}\n")
    print(f"Wrote scored file: {out_path}")


if __name__ == '__main__':
    main()
