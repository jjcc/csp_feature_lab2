#!/usr/bin/env python3
import os, argparse
import numpy as np
import pandas as pd
import joblib

def parse_args():
    ap = argparse.ArgumentParser(description="Score Rescue Model on Tail-Gate-rejected high-reward candidates.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    pack = joblib.load(args.model)
    model = pack["model"]; features = pack["features"]; medians = pack["medians"]; thr = float(pack["threshold"])
    df = pd.read_csv(args.input)
    use = [c for c in features if c in df.columns]
    if not use: raise SystemExit("No overlapping features between model and input.")
    for c in use:
        if c not in df: df[c] = medians.get(c, np.nan)
    X = df[use].copy()
    for c in use: X[c] = X[c].fillna(medians.get(c, np.nan))
    proba = model.predict_proba(X)[:,1] if hasattr(model,"predict_proba") else model.decision_function(X)
    decision = np.where(proba < thr, "RESCUE", "KEEP_REJECTED")
    out = df.copy()
    out["rescue_proba_toxic"] = proba
    out["rescue_threshold"] = thr
    out["rescue_decision"] = decision
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")
    print(out["rescue_decision"].value_counts())

if __name__=="__main__":
    main()
