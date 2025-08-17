import unittest
import joblib
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import time

from pyparsing import Path
from dotenv import load_dotenv
import json

from service.utils import fill_features_with_training_medians, prep_tail_training_df, prep_winner_like_training

load_dotenv()

#out_dir = os.getenv("OUT_DIR", "output")
out_dir = "test/data/output"
#csv_file = "coveredPut_2025-08-13_11_00.csv"
csv_file = "coveredPut_2025-06-12_11_00.csv"
csv_postfix = csv_file.replace("coveredPut_","").replace(".csv","")
csv_path = f"test/data/put/{csv_file}"
#out_path = Path(args.out)
out_path = os.getenv("LABELED_TRADES_WITH_GEX")
out_path = f"{out_dir}/{out_path}_{csv_postfix}"
base_dir = os.getenv("GEX_BASE_DIR")


TAIL_MODEL_IN="models/tail_model_gex_v2_cut05.pkl"
TAIL_KEEP_PROBA_COL="tail_proba"
WINNER_MODEL_IN="output/winner/model_pack.pkl"
WINNER_PROBA_COL = "proba"

def parse_target_time(s: str) -> time:
    try:
        hh, mm = s.split(":")
        return time(int(hh), int(mm))
    except Exception:
        return time(11, 0)

class TestScore(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file to simulate labeled_trades.csv
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = csv_path

        # Create a dummy GEX base dir (empty for now)
        self.gex_base_dir =  base_dir
        self.base_dir = base_dir

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_score_tails_gate(self):
        data_date = "2025-08-13"
        CSV_IN = f"test/data/output/merged_test_{data_date}.csv"
        pack = joblib.load(TAIL_MODEL_IN)
        clf = pack["model"]
        feats = pack["features"]
        med  = pack["medians"]
        raw = pd.read_csv(CSV_IN)
        df = prep_tail_training_df(raw)
        X, medians = fill_features_with_training_medians(df, feats)

        proba = clf.predict_proba(X)[:,1]
        out = df.copy()
        out[TAIL_KEEP_PROBA_COL] = proba
        thresh = 0.04028
        out["is_tail_pred"] = (out[TAIL_KEEP_PROBA_COL] >= thresh).astype(int)
        out_path = f"{out_dir}/scored_tails_test_{data_date}.csv"
        out.to_csv(out_path, index=False)

        # Test evaluating tails gate


    def test_score_winner_classifier(self):
        data_date = "2025-08-12"
        CSV_IN = f"test/data/output/merged_test_{data_date}.csv"

        df = pd.read_csv(CSV_IN)

        pack = joblib.load(WINNER_MODEL_IN)
        clf = pack["model"]
        feats = pack["features"]
        medians = pack.get("medians", None)
        impute_missing = bool(pack.get("impute_missing", bool(medians is not None)))
        Xwc, mask = prep_winner_like_training(df, feats, medians=medians, impute_missing=impute_missing)

        proba = clf.predict_proba(Xwc)[:,1]
        out = df.loc[mask].copy()
        out[WINNER_PROBA_COL] = proba
        thresh = 0.8375
        out["is_winner_pred"] = (out[WINNER_PROBA_COL] >= thresh).astype(int)
        out_path = f"{out_dir}/scored_winner_test_{data_date}.csv"
        out.to_csv(out_path, index=False)

    # Test both
    def test_score_tail_winner_classifier(self):
        data_date = "2025-08-11"
        CSV_IN = f"test/data/output/merged_test_{data_date}.csv"

        df_o = pd.read_csv(CSV_IN)

        pack_tl = joblib.load(TAIL_MODEL_IN)
        clf_tl = pack_tl["model"]
        feats = pack_tl["features"]
        med  = pack_tl["medians"]
        df = prep_tail_training_df(df_o)
        X_tl, medians = fill_features_with_training_medians(df, feats)

        proba = clf_tl.predict_proba(X_tl)[:,1]
        out = df.copy()
        out[TAIL_KEEP_PROBA_COL] = proba
        thresh = 0.04028
        out["is_tail_pred"] = (out[TAIL_KEEP_PROBA_COL] >= thresh).astype(int)


        pack_wc = joblib.load(WINNER_MODEL_IN)
        clf_wc = pack_wc["model"]
        feats = pack_wc["features"]
        medians = pack_wc.get("medians", None)
        impute_missing = bool(pack_wc.get("impute_missing", bool(medians is not None)))
        Xwc, mask = prep_winner_like_training(df, feats, medians=medians, impute_missing=impute_missing)

        proba = clf_wc.predict_proba(Xwc)[:,1]
        out2 = out.loc[mask].copy()
        out2[WINNER_PROBA_COL] = proba
        thresh = 0.8375
        out2["is_winner_pred"] = (out2[WINNER_PROBA_COL] >= thresh).astype(int)
        out_path = f"{out_dir}/scored_tail_winner_test_{data_date}.csv"
        gex_columns = [col for col in out2.columns if col.startswith("gex_")]
        out2.drop(columns=gex_columns, inplace=True, errors='ignore')
        out2.to_csv(out_path, index=False)

if __name__ == '__main__':
    unittest.main()
