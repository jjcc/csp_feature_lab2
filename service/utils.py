#!/usr/bin/env python3
"""
model_utils.py — Shared utilities for the CSP pipeline.

Contains:
- ensure_dir(path): create parent dirs and return path
- env helpers: load_env_default()
- Data prep:
    * prep_tail_like_training(df, feats, medians): uses training medians; fills gex_missing=1
    * prep_winner_like_training(df, feats, medians=None, impute_missing=True): impute or drop-NaN to match training
- Threshold selection:
    * pick_threshold_auto(y_true, proba, targets_prec=None, targets_rec=None)
      - precision target: choose lowest threshold reaching >= target precision
      - recall target: choose highest threshold reaching >= target recall
Additions:
- prep_tail_training_derived(df): EXACTLY replicates training's _prep_df()
  (recompute total_pnl and return_pct; parse datetimes; bid/bidPrice handling)
- fill_features_with_training_medians(df, feat_list, medians): EXACTLY replicates training's _fill_features()
  (create missing cols; gex_missing:=1 when NaN; other NaNs -> training medians)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score


# ---------- FS / ENV helpers ----------

# ---- Existing helpers kept minimal to avoid conflicts ----
def ensure_dir(path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path

def load_env_default() -> None:
    """Try to load .env from CWD; if missing, try alongside the calling script."""
    loaded = load_dotenv()
    if not loaded:
        here = Path(__file__).resolve().parent / ".env"
        if here.exists():
            load_dotenv(dotenv_path=here)


# ---------- Data preparation ----------
def prep_tail_training_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce train_tail_with_gex._prep_df exactly."""
    X = df.copy()

    # Parse datetimes if present
    for c in ("tradeTime","expirationDate"):
        if c in X.columns:
            X[c] = pd.to_datetime(X[c], errors="coerce")

    # Recompute total_pnl and return_pct the same way as trainer
    # If 'bid' missing but 'bidPrice' present, alias it
    if "bid" not in X.columns and "bidPrice" in X.columns:
        X["bid"] = X["bidPrice"]

    bid_eff = X["bid"] if "bid" in X.columns else X.get("bidPrice", pd.Series(index=X.index, dtype=float))

    X["entry_credit"] = pd.to_numeric(bid_eff, errors="coerce").fillna(0.0) * 100.0

    strike = pd.to_numeric(X["strike"], errors="coerce")
    expiry_close = pd.to_numeric(X.get("expiry_close"), errors="coerce")
    X["exit_intrinsic"] = np.maximum(strike - expiry_close, 0.0) * 100.0

    X["total_pnl"] = X["entry_credit"] - X["exit_intrinsic"]

    capital = pd.to_numeric(X["strike"], errors="coerce") * 100.0
    X["return_pct"] = 100.0 * X["total_pnl"] / capital

    return X

def fill_features_with_training_medians(df: pd.DataFrame, feat_list: List[str], medians: Dict[str, float]) -> pd.DataFrame:
    """
    Reproduce train_tail_with_gex._fill_features for scoring:
    - Ensure all features exist
    - For 'gex_missing': fillna(1)
    - For all others: fillna(training_median)
    """
    X = df.copy()
    for c in feat_list:
        if c not in X.columns:
            X[c] = np.nan
        if c == "gex_missing":
            X[c] = X[c].fillna(1)
        else:
            med = float(medians.get(c, 0.0))
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med)

    return X[feat_list].astype(float)


def prep_winner_like_training(
    df: pd.DataFrame,
    feats: List[str],
    medians: Optional[Dict[str, float]] = None,
    impute_missing: bool = True,
):
    """
    Prepare features like winner training.
    Returns (X_prepared, mask_kept_rows). If impute_missing=False, drops rows with any NaNs.
    """
    if impute_missing:
        X = df.copy()
        for c in feats:
            if c not in X.columns:
                X[c] = np.nan
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna((medians or {}).get(c, 0.0))
        mask = pd.Series(True, index=df.index)
        return X[feats], mask
    else:
        X = df[feats].apply(pd.to_numeric, errors="coerce")
        mask = (~X.isna().any(axis=1))
        return X[mask], mask


# ---------- Threshold selection ----------

def pick_threshold_auto(
    y_true: np.ndarray,
    proba: np.ndarray,
    targets_prec: Optional[List[float]] = None,
    targets_rec: Optional[List[float]] = None,
):
    """Return (chosen_threshold, pandas.DataFrame row) according to constraints.
       - Precision target: choose the *lowest threshold* achieving >= first target precision.
       - Recall target: choose the *highest threshold* achieving >= first target recall.
       If neither applies, returns (None, None).
    """
    import pandas as pd  # local import to avoid global dependency here

    if targets_prec:
        targets_prec = [float(targets_prec[0])]
    if targets_rec:
        targets_rec = [float(targets_rec[0])]

    if not targets_prec and not targets_rec:
        return None, None

    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    def metrics_at(thr: float):
        yhat = (proba >= thr).astype(int)
        return dict(
            precision=precision_score(y_true, yhat, zero_division=0),
            recall=recall_score(y_true, yhat, zero_division=0),
            f1=f1_score(y_true, yhat, zero_division=0),
            coverage=float(yhat.mean()),
            threshold=float(thr),
        )

    if targets_prec:
        t = targets_prec[0]
        chosen = None
        for thr in sorted(thresholds):
            m = metrics_at(thr)
            if m["precision"] >= t:
                chosen = m; break
        if chosen is None and len(thresholds):
            chosen = metrics_at(float(max(thresholds)))
        row = pd.DataFrame([{**chosen, "target_type": "precision", "target": float(t)}])
        return chosen["threshold"], row

    if targets_rec:
        t = targets_rec[0]
        chosen = None
        for thr in sorted(thresholds, reverse=True):
            m = metrics_at(thr)
            if m["recall"] >= t:
                chosen = m
        if chosen is None and len(thresholds):
            chosen = metrics_at(float(min(thresholds)))
        row = pd.DataFrame([{**chosen, "target_type": "recall", "target": float(t)}])
        return chosen["threshold"], row

    return None, None
