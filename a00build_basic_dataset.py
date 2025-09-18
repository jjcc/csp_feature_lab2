
#!/usr/bin/env python3
"""
CSP Feature Lab
---------------
Iterate all CSP snapshot CSVs in a folder, label outcomes at expiry, and run a
RandomForest feature-importance analysis to see which parameters relate to profitability.

Usage:
    pip install -r requirements.txt
    python analyze_csp_features.py --data_dir /path/to/folder --glob "coveredPut_*.csv" --max_rows 0

Outputs (in the same folder as this script):
    labeled_trades.csv        - all labeled rows (one per option row that could be labeled)
    feature_importances.csv   - sorted feature importances from RandomForest
    classification_report.txt - train/test metrics
"""

import os
from os import getenv
from service.preprocess import load_csp_files 

from dotenv import load_dotenv
load_dotenv()
def main():

    data_dir = getenv("DATA_DIR", "")
    glob_pat = getenv("GLOB", "coveredPut_*.csv")
    target_time = getenv("TARGET_TIME", "11:00")

    raw = load_csp_files(data_dir, glob_pat, target_time=target_time, enforce_daily_pick=True)
    #raw = load_csp_files(data_dir, glob_pat, target_time=target_time, enforce_daily_pick=False)
    raw = raw.drop(columns=["baseSymbolType","Unnamed: 0", "symbolType"], errors='ignore')
    # rename index to "row_id" for tracking
    raw = raw.reset_index().rename(columns={"index": "row_id"})

    basic_csv = getenv("BASIC_CSV", "labeled_trades_normal.csv")
    out_dir = getenv("OUTPUT_DIR", "./output")
    raw.to_csv(os.path.join(out_dir, basic_csv), index=False)


if __name__ == "__main__":
    main()
