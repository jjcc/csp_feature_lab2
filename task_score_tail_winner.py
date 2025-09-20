#!/usr/bin/env python3
"""Score tail + winner pipeline as a standalone script.

This is a converted version of the test `test_score_tail_winner_classifier` from
`test/test_prod.py` made into a runnable script placed at the repository root.
"""
from glob import glob
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime, time

from dotenv import load_dotenv

from service.data_prepare import _load_vix, add_macro_features
from service.get_vix import get_current_vix, init_driver, url_vix
from service.preprocess import merge_gex
from service.utils import fill_features_with_training_medians, prep_tail_training_df, prep_winner_like_training

load_dotenv()

#TAIL_MODEL_IN = "models/tail_model_gex_v2_cut05.pkl"
TAIL_MODEL_IN = "output/tails_train/v6b_ne/tail_model_gex_v6b_ne_cut05.pkl"
TAIL_KEEP_PROBA_COL = "tail_proba"
#WINNER_MODEL_IN = "output/winner_train/v6_oof_ne_straited_w_lgbm/winner_classifier_v6_oof_ne_w_lgbm.pkl"
#WINNER_MODEL_IN = "output/winner_train/external/winner_classifier_v6_oof_ne_w_lgbm.pkl"
WINNER_MODEL_IN = "output/winner_train/v6_oof_ne_ts_w_lgbm_rfctr/winner_classifier_v6_oof_ne_w_lgbm.pkl"
WINNER_PROBA_COL = "winner_proba"

PX_BASE_DIR = os.getenv("PX_BASE_DIR", "").strip()  

def parse_target_time(s: str) -> time:
    try:
        hh, mm = s.split(":")
        return time(int(hh), int(mm))
    except Exception:
        return time(11, 0)


def get_merge_params(ignore_log=False):
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
    if not ignore_log and latest_file in files:
        print(f"Latest file {latest_file} already processed.")
        return None, None
    # get time of latest_file, like coveredPut_2025-08-15_16_30.csv
    latest_file_time = latest_file.split("_")[-2:]
    latest_file_time = ":".join(latest_file_time).replace(".csv", "")
    target_t = parse_target_time(latest_file_time)
    target_minutes = target_t.hour * 60 + target_t.minute
    return latest_file_with_path, target_minutes


def main(Test=False):
    latest_file_with_path, target_minutes = get_merge_params(True)
    if latest_file_with_path is None:
        print("File is processed or no file found.")
        return

    option_file = latest_file_with_path
    if Test:
        #option_file = "option/put/unprocessed/coveredPut_2025-08-08_11_00.csv"
        option_file = "option/put/coveredPut_2025-09-18_11_00.csv"
        latest_file_time = option_file.split("_")[-2:]
        latest_file_time = ":".join(latest_file_time).replace(".csv", "")
        target_t = parse_target_time(latest_file_time)
        target_minutes = target_t.hour * 60 + target_t.minute
        #target_minutes = 16 * 60 + 30

    # construct an output file path
    hour, minute = option_file.replace(".csv", "").split("_")[-2:]
    target_date = option_file.split("_")[1]
    time_str = f"{hour}_{minute}"
    out_dir = "prod/output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/scored_tail_winner_lgbm_{target_date}_{time_str}.csv"
    if Test:
        out_path = f"{out_dir}/scored_tail_winner_lgbm_{target_date}_{time_str}_test2.csv"

    df_o = add_features(target_minutes, option_file, target_date)

    # end of add macro features
    # get next earning and previous earning




    # tail scoring
    #pack_tl = joblib.load(TAIL_MODEL_IN)
    #clf_tl = pack_tl["model"]
    #feats = pack_tl["features"]
    #med = pack_tl.get("medians", None)
    df = prep_tail_training_df(df_o)
    #X_tl, medians = fill_features_with_training_medians(df, feats)

    #proba = clf_tl.predict_proba(X_tl)[:, 1]
    out = df.copy()
    #out[TAIL_KEEP_PROBA_COL] = proba
    #thresh = 0.03
    #out["is_tail_pred"] = (out[TAIL_KEEP_PROBA_COL] >= thresh).astype(int)

    # winner scoring
    pack_wc = joblib.load(WINNER_MODEL_IN)
    clf_wc = pack_wc["model"]
    feats = pack_wc["features"]
    medians = pack_wc.get("medians", None)
    impute_missing = bool(pack_wc.get("impute_missing", bool(medians is not None)))
    Xwc, mask = prep_winner_like_training(out, feats, medians=medians, impute_missing=impute_missing)

    proba = clf_wc.predict_proba(Xwc)[:, 1]
    out2 = out.loc[mask].copy()
    out2[WINNER_PROBA_COL] = proba
    thresh = 0.95 # 
    out2["is_winner_pred"] = (out2[WINNER_PROBA_COL] >= thresh).astype(int)

    # cleanup and filters
    gex_columns = [col for col in out2.columns if col.startswith("gex_")]
    out2.drop(columns=gex_columns, inplace=True, errors='ignore')
    to_drop = [
        'baseSymbolType', 'expirationDate',
        'strike', 'moneyness', 'breakEvenBid', 'percentToBreakEvenBid', 'tradeTime', 'symbol_norm', 'impliedVolatilityRank1y',
        'delta', 'breakEvenProbability', 'expirationType', 'symbolType',
        'entry_credit', 'exit_intrinsic', 'total_pnl', 'return_pct',
        'ret_2d', 'ret_5d', 'ret_2d_norm', 'ret_5d_norm','prev_close','prev_close_minus_ul','prev_close_minus_ul_pct',
        'log1p_DTE','bid'
    ]
    out2.drop(columns=to_drop, inplace=True, errors='ignore')
    #out2["verdict"] = (out2["is_winner_pred"] == 1) & (out2["is_tail_pred"] == 0)
    #out2 = out2[out2["is_winner_pred"] == 1]
    #out2 = out2[out2["is_tail_pred"] == 0]
    out2.to_csv(out_path, index=False)

    # log processing information
    process_log = "log/processed.log"
    latest_file = os.path.basename(latest_file_with_path)
    with open(process_log, "a") as f:
        f.write(f"{latest_file}\n")
    print(f"Wrote scored file: {out_path}")

def add_features(target_minutes, option_file, target_date):
    gex_base_dir = os.getenv("GEX_BASE_DIR")
    trades = pd.read_csv(option_file)
    df_o = merge_gex(trades, gex_base_dir, target_minutes=target_minutes)

    # add macro features using shared function
    today = datetime.now()
    # get VIX (handles both real-time and historic)
    vix_df = get_vix(today, target_date=datetime.strptime(target_date, "%Y-%m-%d").date())

    # Use shared macro features function with pre-built VIX DataFrame
    df_o = add_macro_features(df_o, vix_df, PX_BASE_DIR)
    return df_o

def get_vix(today, target_date=None):
    if target_date is not None:
        if target_date != today.date():
            VIX_CSV = os.getenv("VIX_CSV", "").strip() or None
            start_date = target_date - pd.Timedelta(days=1)
            end_date   = target_date + pd.Timedelta(days=1)
            vix = _load_vix(VIX_CSV,start_date, end_date)
            vix_df = pd.DataFrame({"trade_date": vix.index, "VIX": vix.values})
            print(f"Target date {target_date} is not today {today.date()}, skip VIX fetch.")
            return vix_df

    driver = init_driver(headless=True)
    vix_value = get_current_vix(url_vix, driver)
    try:
        vix_value = float(vix_value)
        today_date = today.date()
        today_date = pd.to_datetime(today_date)
    except ValueError:
        vix_value = 16.35
    vix_df = pd.DataFrame({"trade_date": [today_date], "VIX": [vix_value]})
    return vix_df


if __name__ == '__main__':
    #main(Test=True)
    main()
