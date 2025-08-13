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

def build_X(df, features, medians, scaler):
    Xcols = {}
    for c in features:
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce")
            fill = medians.get(c)
            if fill is None or (isinstance(fill, float) and np.isnan(fill)):
                med = pd.to_numeric(df[c], errors="coerce").median()
                fill = 0.0 if np.isnan(med) else float(med)
            Xcols[c] = col.fillna(fill).astype(float)
        else:
            Xcols[c] = pd.Series(0.0, index=df.index)
    X = pd.DataFrame(Xcols)[features].values
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass
    return X

def main():
    #env_file = os.environ.get("ENV_FILE", "score_tail.env")
    env_file = os.environ.get("ENV_FILE", ".env_score_tail")
    if os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        load_dotenv()

    CSV = os.getenv("CSV", "labeled_trades_with_gex.csv")
    MODEL = os.getenv("MODEL", "tail_model_gex_v1b.pkl")
    OUT_SCORES = os.getenv("OUT_SCORES", "tail_gex_p35_scores.csv")

    df = pd.read_csv(CSV)
    pack = infer_pack(load_pickle(MODEL))
    model, feats = pack["model"], pack["features"]
    if not feats:
        print("[ERROR] Tail model missing features metadata].", file=sys.stderr); sys.exit(1)
    if not hasattr(model, "predict_proba"):
        print("[ERROR] Tail model lacks predict_proba()", file=sys.stderr); sys.exit(1)

    X = build_X(df, feats, pack["medians"], pack["scaler"])
    proba = model.predict_proba(X)[:,1]

    out = pd.DataFrame({"score": proba})
    for k in ["trade_id","baseSymbol","tradeTime","expirationDate","strike"]:
        if k in df.columns: out[k] = df[k].values
    out.to_csv(OUT_SCORES, index=False)
    print(json.dumps({"rows": int(len(out)), "out_scores": OUT_SCORES}, indent=2))

if __name__ == "__main__":
    main()
