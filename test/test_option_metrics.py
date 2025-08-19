from glob import glob
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

from service.option_metrics import compute_option_metrics
from service.preprocess import merge_gex, pick_daily_snapshot_files
from service.utils import fill_features_with_training_medians, prep_tail_training_df, prep_winner_like_training

load_dotenv()


class TestOptionMetrics(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_get_metrics(self):
        option_data_dir = "option/put"
        glob_pat = f"coveredPut_*.csv"
        # get the latest file
        latest_file_with_path = max(glob(os.path.join(option_data_dir, glob_pat)), key=os.path.getctime)
        a_file = latest_file_with_path

        df_o = pd.read_csv(a_file)
        df = df_o[["underlyingLastPrice", "strike", "bidPrice", "daysToExpiration"]].copy()

        # If your file *doesn't* contain impliedVolatility (only IV Rank), skip probabilities:
        out = compute_option_metrics(df, iv_col="impliedVolatility", add_probabilities=False)

        # Inspect the enriched DataFrame
        print(out.head())
        out.to_csv("test/data/output/with_metrics.csv", index=False)

 

if __name__ == '__main__':
    unittest.main()
