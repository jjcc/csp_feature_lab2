
#!/usr/bin/env python3
# Merge GEX features into labeled_trades.csv
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from datetime import time
from pathlib import Path
from dotenv import load_dotenv
from service.preprocess import  merge_gex


def parse_target_time(s: str) -> time:
    try:
        hh, mm = s.split(":")
        return time(int(hh), int(mm))
    except Exception:
        return time(11, 0)



def main():
    load_dotenv()
    #ap = argparse.ArgumentParser()
    #ap.add_argument("--csv", default="labeled_trades.csv", help="Path to labeled_trades.csv")
    #ap.add_argument("--out", default="labeled_trades_with_gex.csv", help="Output CSV path")
    #args = ap.parse_args()

    out_dir = os.getenv("OUT_DIR", "output")
    csv_input = os.getenv("BASIC_CSV")
    csv_path = f"{out_dir}/{csv_input}"
    #out_path = Path(args.out)
    out_path = os.getenv("GEX_CSV")
    out_path = f"{out_dir}/{out_path}"

    base_dir = os.getenv("GEX_BASE_DIR")
    target_time_str = os.getenv("GEX_TARGET_TIME", "11:00")
    if not base_dir:
        raise SystemExit("GEX_BASE_DIR is not set in .env")

    target_t = parse_target_time(target_time_str)
    target_minutes = target_t.hour * 60 + target_t.minute

    trades = pd.read_csv(csv_path)
    merged = merge_gex(trades, base_dir, target_minutes)

    merged.to_csv(out_path, index=False)

    rep = {
        "rows": len(merged),
        "gex_found": int((merged["gex_missing"] == 0).sum()),
        "gex_missing": int((merged["gex_missing"] == 1).sum()),
        "base_dir": base_dir,
        "target_time": target_time_str
    }
    import time as t
    today = t.strftime("%Y%m%d")
    with open(f"{out_dir}/merge_gex_report_{today}.json","w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
