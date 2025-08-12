
#!/usr/bin/env python3
"""
Score trades with the tail-loss model and gate by a fixed threshold (.env-driven).

- Loads MODEL_IN (joblib with model, medians, features)
- Scores CSV_INPUT and writes SCORED_OUT with tail_proba and tail_flag
- Applies THRESHOLD to produce tail_flag (1=rejected, 0=pass)
- If THRESHOLD_FROM is set (best_f1, recall_95, etc.), it can override THRESHOLD
  when paired with METRICS_IN (not required here).

ENV
---
CSV_INPUT=path/to/labeled_trades_with_gex.csv
MODEL_IN=path/to/tail_model_gex_v1.pkl
SCORED_OUT=./scored_with_tail.csv
THRESHOLD=0.086907
# Optional:
# THRESHOLD_FROM=(best_f1|recall_95|recall_98)
# METRICS_IN=path/to/metrics.json
"""
import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path

# Robust .env loading
def _load_env():
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    loaded = load_dotenv()
    if not loaded:
        script_env = Path(__file__).resolve().parent / ".env"
        if script_env.exists():
            load_dotenv(dotenv_path=script_env)
_load_env()

CSV_INPUT = os.getenv("CSV_INPUT", "/mnt/data/labeled_trades_with_gex.csv")
MODEL_IN = os.getenv("MODEL_IN", "/mnt/data/tail_model_gex_v1.pkl")
SCORED_OUT = os.getenv("SCORED_OUT", "/mnt/data/scored_with_tail.csv")
THRESHOLD = os.getenv("THRESHOLD")
THRESHOLD_FROM = os.getenv("THRESHOLD_FROM", "").strip().lower()
METRICS_IN = os.getenv("METRICS_IN", "")

def _resolve_threshold():
    if THRESHOLD_FROM and METRICS_IN and Path(METRICS_IN).exists():
        with open(METRICS_IN) as f:
            m = json.load(f)
        if THRESHOLD_FROM == "best_f1":
            return float(m.get("oof_best_f1_threshold", m.get("oof_best_threshold", 0.5)))
        if THRESHOLD_FROM == "recall_95":
            return float(m.get("thr_recall_95", 0.2))
        if THRESHOLD_FROM == "recall_98":
            return float(m.get("thr_recall_98", 0.15))
    if THRESHOLD is None:
        raise SystemExit("THRESHOLD not set. Provide THRESHOLD or use THRESHOLD_FROM with METRICS_IN.")
    return float(THRESHOLD)

def main():
    art = joblib.load(MODEL_IN)
    model = art["model"]
    medians = art["medians"]
    feats = art["features"]

    df = pd.read_csv(CSV_INPUT)
    # Ensure required columns exist and impute like training
    for c in feats:
        if c not in df.columns:
            df[c] = np.nan
        if c == "gex_missing":
            df[c] = df[c].fillna(1)
        else:
            df[c] = df[c].fillna(medians.get(c, 0.0))

    X = df[feats].astype(float).values
    proba = model.predict_proba(X)[:, 1]

    thr = _resolve_threshold()

    out = df.copy()
    out["tail_proba"] = proba
    out["tail_flag"] = (out["tail_proba"] >= thr).astype(int)  # 1 = reject

    Path(SCORED_OUT).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SCORED_OUT, index=False)
    print(f"[OK] wrote {SCORED_OUT} with threshold={thr}")

if __name__ == "__main__":
    main()
