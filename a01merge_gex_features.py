
#!/usr/bin/env python3
# Merge GEX features into labeled_trades.csv
import os
import re
import json
import math
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import time
from pathlib import Path
from dotenv import load_dotenv

FILENAME_RE = re.compile(r'^(?P<symbol>[A-Za-z0-9.\-_]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})-(?P<minute>\d{2})\.csv$')

def parse_target_time(s: str) -> time:
    try:
        hh, mm = s.split(":")
        return time(int(hh), int(mm))
    except Exception:
        return time(11, 0)

def parse_gex_filename(name: str):
    m = FILENAME_RE.match(name)
    if not m:
        return None
    dd = m.group('date')
    hh = int(m.group('hour')); mm = int(m.group('minute'))
    return {
        "symbol": m.group('symbol').lower(),
        "date": dd,
        "stamp": f"{hh:02d}:{mm:02d}",
        "minutes": hh * 60 + mm,
    }

def pick_closest_file(files, target_minutes: int):
    best = None; best_dist = 10**9
    for p in files:
        meta = parse_gex_filename(os.path.basename(p))
        if not meta: 
            continue
        dist = abs(meta["minutes"] - target_minutes)
        if dist < best_dist:
            best = p; best_dist = dist
    return best

def compute_gex_features(df_gex: pd.DataFrame, ul_price: float) -> dict:
    out = {
        "gex_total": np.nan,
        "gex_total_abs": np.nan,
        "gex_pos": np.nan,
        "gex_neg": np.nan,
        "gex_center_abs_strike": np.nan,
        "gex_flip_strike": np.nan,
        "gex_gamma_at_ul": np.nan,
        "gex_distance_to_flip": np.nan,
        "gex_sign_at_ul": np.nan,
    }
    if df_gex is None or df_gex.empty:
        return out

    g = df_gex.copy()
    if "strike" not in g.columns or "gamma" not in g.columns:
        return out
    g = g[["strike", "gamma"]].dropna()
    if g.empty:
        return out
    g["strike"] = pd.to_numeric(g["strike"], errors="coerce")
    g["gamma"] = pd.to_numeric(g["gamma"], errors="coerce")
    g = g.dropna()
    if g.empty:
        return out
    g = g.sort_values("strike")

    total = g["gamma"].sum()
    total_abs = g["gamma"].abs().sum()
    pos = g.loc[g["gamma"] > 0, "gamma"].sum()
    neg = g.loc[g["gamma"] < 0, "gamma"].sum()

    out["gex_total"] = float(total)
    out["gex_total_abs"] = float(total_abs)
    out["gex_pos"] = float(pos)
    out["gex_neg"] = float(neg)

    if total_abs > 0:
        out["gex_center_abs_strike"] = float((g["strike"] * g["gamma"].abs()).sum() / total_abs)

    cum = g["gamma"].cumsum().values
    strikes = g["strike"].values
    flip = np.nan
    for i in range(1, len(cum)):
        if np.sign(cum[i-1]) != np.sign(cum[i]):
            x1, y1 = strikes[i-1], cum[i-1]
            x2, y2 = strikes[i], cum[i]
            if (y2 - y1) != 0:
                t = -y1 / (y2 - y1)
                flip = x1 + t * (x2 - x1)
            else:
                flip = strikes[i]
            break
    out["gex_flip_strike"] = float(flip) if not (isinstance(flip, float) and math.isnan(flip)) else np.nan

    if ul_price is not None and not np.isnan(ul_price):
        try:
            out["gex_gamma_at_ul"] = float(np.interp(ul_price, strikes, g["gamma"].values))
            out["gex_sign_at_ul"] = float(np.sign(out["gex_gamma_at_ul"]))
        except Exception:
            pass

    if pd.notna(out["gex_flip_strike"]) and ul_price is not None and not np.isnan(ul_price):
        out["gex_distance_to_flip"] = float(out["gex_flip_strike"] - ul_price)

    return out

def main():
    load_dotenv()
    #ap = argparse.ArgumentParser()
    #ap.add_argument("--csv", default="labeled_trades.csv", help="Path to labeled_trades.csv")
    #ap.add_argument("--out", default="labeled_trades_with_gex.csv", help="Output CSV path")
    #args = ap.parse_args()

    out_dir = os.getenv("OUT_DIR", "output")
    csv_path = f"{out_dir}/labeled_trades.csv"
    #out_path = Path(args.out)
    out_path = os.getenv("LABELED_TRADES_WITH_GEX")
    out_path = f"{out_dir}/{out_path}"

    base_dir = os.getenv("GEX_BASE_DIR")
    target_time_str = os.getenv("GEX_TARGET_TIME", "11:00")
    if not base_dir:
        raise SystemExit("GEX_BASE_DIR is not set in .env")

    target_t = parse_target_time(target_time_str)
    target_minutes = target_t.hour * 60 + target_t.minute

    trades = pd.read_csv(csv_path)
    need = ["baseSymbol","tradeTime","underlyingLastPrice"]
    for c in need:
        if c not in trades.columns:
            raise SystemExit(f"Missing required column: {c}")
    trades["tradeTime"] = pd.to_datetime(trades["tradeTime"], errors="coerce")
    trades["trade_date"] = trades["tradeTime"].dt.strftime("%Y-%m-%d")
    trades["symbol_norm"] = trades["baseSymbol"].astype(str).str.lower()

    cache = {}
    feats = []

    for idx, row in trades.iterrows():
        sym = row["symbol_norm"]
        d = row["trade_date"]
        ul = float(row["underlyingLastPrice"]) if pd.notna(row["underlyingLastPrice"]) else np.nan
        key = (sym, d)
        if key not in cache:
            day_dir = Path(base_dir) / d
            if not day_dir.exists():
                cache[key] = (None, None)
            else:
                pattern = str(day_dir / f"{sym}_{d}_*.csv")
                cand = glob.glob(pattern)
                if cand:
                    chosen = pick_closest_file(cand, target_minutes)
                    try:
                        gex_df = pd.read_csv(chosen)
                    except Exception:
                        gex_df = None
                    cache[key] = (gex_df, chosen)
                else:
                    cache[key] = (None, None)

        gex_df, chosen_path = cache[key]
        feat = compute_gex_features(gex_df, ul)
        feat["gex_file"] = chosen_path if chosen_path else ""
        feat["gex_missing"] = 0 if gex_df is not None and not gex_df.empty else 1
        feats.append(feat)

    gex_df_all = pd.DataFrame(feats, index=trades.index)
    merged = pd.concat([trades, gex_df_all], axis=1)
    merged.to_csv(out_path, index=False)

    rep = {
        "rows": len(merged),
        "gex_found": int((merged["gex_missing"] == 0).sum()),
        "gex_missing": int((merged["gex_missing"] == 1).sum()),
        "base_dir": base_dir,
        "target_time": target_time_str
    }
    with open(f"{out_dir}/merge_gex_report2.json","w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
