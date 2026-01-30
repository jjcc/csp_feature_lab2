#!/usr/bin/env python3
"""Apply a saved rescue model pack to a dataset.

Loads a joblib pack produced by `train_rescue_model.py`, fills missing features
with medians from the pack, computes P(toxic) and flags rows to "rescue"
(proba < threshold).

Usage examples:
  export RESCUE_PACK=output/rescue_tail/rescue_model.joblib
  python apply_rescue.py --input data/new_candidates.csv --out scored.csv

Options:
  --input     Path to input CSV (required)
  --model     Path to rescue_model.joblib (defaults to env RESCUE_PACK)
  --out       Path to output CSV (defaults to stdout if not provided)
  --pred-col  Column name to write predicted proba (default: rescue_proba_toxic)
  --flag-col  Column name to write rescue decision (default: is_rescue)
  --filter    If provided, apply pack['selection_query'] to input before scoring

The script is defensive: it will add missing feature columns, impute with medians
from the pack, and handle models that either expose predict_proba or decision_function.
"""
import os
import argparse
import json
import sys
import joblib
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Apply rescue model pack to CSV rows")
    p.add_argument("--input", required=True, help="Input CSV file path")
    p.add_argument("--model", default=os.getenv("RESCUE_PACK"), help="Rescue model joblib pack (env RESCUE_PACK)")
    p.add_argument("--out", default=None, help="Output CSV path (default: stdout)")
    p.add_argument("--pred-col", default="rescue_proba_toxic", help="Column for predicted P(toxic)")
    p.add_argument("--flag-col", default="is_rescue", help="Column for rescue boolean (proba < threshold)")
    p.add_argument("--filter", action="store_true", help="Apply pack['selection_query'] to input before scoring if present")
    return p.parse_args()


def safe_predict_proba(model, X):
    """Return a 1-d array of probabilities for the positive label (toxic).
    If predict_proba exists, use [:,1]. Otherwise, use decision_function and
    map to probability-like scores via logistic transform (approx).
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # If binary, take column 1
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        # fallback: return first column
        return proba.ravel()
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # convert scores to [0,1] via logistic
        try:
            from scipy.special import expit
            return expit(scores)
        except Exception:
            return 1.0/(1.0 + np.exp(-scores))
    else:
        raise RuntimeError("Model has neither predict_proba nor decision_function")


def main():
    args = parse_args()
    if not args.model:
        raise SystemExit("Model path not provided; set --model or RESCUE_PACK env var")
    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")
    pack = joblib.load(args.model)
    model = pack.get("model")
    features = pack.get("features", [])
    medians = pack.get("medians", {})
    threshold = float(pack.get("threshold", pack.get("thresh", 0.5)))

    df = pd.read_csv(args.input)

    # Optionally filter by selection_query from pack
    if args.filter and pack.get("selection_query"):
        try:
            sel = pack.get("selection_query")
            df = df.query(sel).copy()
        except Exception as e:
            print(f"[WARN] could not apply selection_query='{pack.get('selection_query')}' - {e}")

    # Ensure features exist; add missing columns as NaN
    for f in features:
        if f not in df.columns:
            df[f] = np.nan

    # Reorder X according to features
    X = df[features].copy()

    # Impute using medians from pack (fall back to column median)
    for c in features:
        if X[c].isna().any():
            if c in medians and medians[c] is not None:
                X[c] = X[c].fillna(medians[c])
            else:
                X[c] = X[c].fillna(X[c].median())

    # If the model is a pipeline that expects 2D numpy, pass DataFrame as-is
    try:
        proba = safe_predict_proba(model, X)
    except Exception as e:
        raise SystemExit(f"Prediction failed: {e}")

    # attach predictions
    df[args.pred_col] = proba
    df[args.flag_col] = (df[args.pred_col] < threshold).astype(int)

    # Write output
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Wrote scored file: {args.out}")
    else:
        # stream to stdout
        df.to_csv(sys.stdout, index=False)


if __name__ == '__main__':
    main()
