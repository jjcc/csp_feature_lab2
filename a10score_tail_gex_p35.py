#!/usr/bin/env python3
import os, sys, json, math
import numpy as np
import pandas as pd
from dotenv import load_dotenv

def load_pickle(path):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

def infer_pack(obj):
    if isinstance(obj, dict):
        return {
            "model": obj.get("model", obj),
            "features": obj.get("features"),
            "medians": obj.get("medians", {}) or {},
            "scaler": obj.get("scaler"),
        }
    feats = getattr(obj, "feature_names_in_", None)
    return {"model": obj, "features": list(feats) if feats is not None else None, "medians": {}, "scaler": None}

def safe_numeric(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s

def build_X(df, features, medians, scaler):
    Xcols = {}
    for c in features:
        if c in df.columns:
            if c == "gex_missing":
                col = (col.fillna(1)).astype(float)
                Xcols[c] = col
            else:
                col = safe_numeric(df[c])
                # Choose fill value priority: training-median -> input-median -> 0.0
                fill = medians.get(c)
                if fill is None or (isinstance(fill, float) and np.isnan(fill)):
                    med = col.median()
                    fill = 0.0 if np.isnan(med) else float(med)
                Xcols[c] = col.fillna(fill).astype(float)
        else:
            # Column missing entirely → use training median if present, else 0.0
            fill = medians.get(c, 0.0)
            if fill is None or (isinstance(fill, float) and np.isnan(fill)):
                fill = 0.0
            Xcols[c] = pd.Series(fill, index=df.index, dtype=float)
    X = pd.DataFrame(Xcols)[features].values
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            print(f"[WARN] scaler.transform failed: {e}", file=sys.stderr)
    return X

def main():
    #env_file = os.environ.get("ENV_FILE", "score_tail.env")
    env_file = os.environ.get("ENV_FILE", ".env_score_tail")
    if os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        load_dotenv()

    CSV = os.getenv("CSV_X", "labeled_trades_with_gex.csv")
    #CSV =  "test/data/output/merged_test.csv"
    MODEL = os.getenv("MODEL", "tail_model_gex_v1b.pkl")
    OUT_SCORES = os.getenv("OUT_SCORES_X", "tail_gex_p35_scores.csv")

    df = pd.read_csv(CSV)
    pack = infer_pack(load_pickle(MODEL))
    model, feats, medians, scaler = pack["model"], pack["features"], pack["medians"], pack["scaler"]

    if not feats:
        print("[ERROR] Tail model missing features metadata].", file=sys.stderr); sys.exit(1)
    if not hasattr(model, "predict_proba"):
        print("[ERROR] Tail model lacks predict_proba()", file=sys.stderr); sys.exit(1)

    X = build_X(df, feats, medians, scaler)
    proba = model.predict_proba(X)[:,1]

    out = pd.DataFrame({"score": proba})
    for k in ["trade_id","baseSymbol","tradeTime","expirationDate","strike"]:
        if k in df.columns: out[k] = df[k].values
    out.to_csv(OUT_SCORES, index=False)
    #out.to_csv("test/data/output/tail_gex_scores_test.csv", index=False)
    print(json.dumps({"rows": int(len(out)), "out_scores": OUT_SCORES}, indent=2))
    #print(json.dumps({"rows": int(len(out)), "out_scores": "test/data/output/tail_gex_scores_test.csv"}, indent=2))

if __name__ == "__main__":
    main()
