#!/usr/bin/env python3
"""Score tail + winner pipeline as a standalone script.

This is a converted version of the test `test_score_tail_winner_classifier` from
`test/test_prod.py` made into a runnable script placed at the repository root.
"""
from glob import glob
import os

from dotenv import load_dotenv

from service.utils import prep_tail_training_df
from service.winner_scoring import load_winner_model, score_winner_data, apply_winner_threshold, cleanup_columns_for_production
from service.production_data import add_features, parse_target_time
from service.env_config import getenv

load_dotenv()

#TAIL_MODEL_IN = "models/tail_model_gex_v2_cut05.pkl"
TAIL_MODEL_IN = "output/tails_train/v6b_ne/tail_model_gex_v6b_ne_cut05.pkl"
TAIL_KEEP_PROBA_COL = "tail_proba"
#WINNER_MODEL_IN = "output/winner_train/v6_oof_ne_straited_w_lgbm/winner_classifier_v6_oof_ne_w_lgbm.pkl"
#WINNER_MODEL_IN = "output/winner_train/external/winner_classifier_v6_oof_ne_w_lgbm.pkl"
#WINNER_MODEL_IN = "output/winner_train/v6_oof_ne_ts_w_lgbm_rfctr/winner_classifier_v6_oof_ne_w_lgbm.pkl"
WINNER_MODEL_IN = "output/winner_train/v7_oof_ne_ts_w_lgbm_tr_ts/winner_classifier_v7_lgbm.pkl"
WINNER_PROBA_COL = "winner_proba"

PX_BASE_DIR = getenv("MACRO_PX_BASE_DIR", "").strip()  



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
        option_file = "option/put/coveredPut_2025-09-29_15_00.csv"
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

    # winner scoring using shared functions
    model_pack = load_winner_model(WINNER_MODEL_IN)
    out2, _, _ = score_winner_data(out, model_pack, WINNER_PROBA_COL)

    thresh = 0.95
    out2 = apply_winner_threshold(out2, WINNER_PROBA_COL, "is_winner_pred", thresh)

    # cleanup and filters using shared function
    out2 = cleanup_columns_for_production(out2)
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



if __name__ == '__main__':
    main(Test=True)
    #main()
