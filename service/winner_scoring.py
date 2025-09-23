#!/usr/bin/env python3
"""
Shared functionality for winner classifier scoring.

This module contains common functions used by both:
- score_winner_classifier_env.py (evaluation/training)
- task_score_tail_winner.py (production scoring)
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from sklearn.metrics import average_precision_score, roc_auc_score
from .utils import prep_winner_like_training, pick_threshold_auto


@dataclass
class WinnerModelPack:
    """Container for loaded winner model and metadata."""
    model: Any
    features: List[str]
    medians: Optional[Dict[str, float]]
    impute_missing: bool
    best_f1_threshold: float
    metrics: Optional[Dict[str, Any]] = None


def load_winner_model(model_path: str) -> WinnerModelPack:
    """Load winner classifier model from pickle file."""
    pack = joblib.load(model_path)

    return WinnerModelPack(
        model=pack["model"],
        features=pack["features"],
        medians=pack.get("medians", None),
        impute_missing=bool(pack.get("impute_missing", bool(pack.get("medians") is not None))),
        best_f1_threshold=float(pack.get("metrics", {}).get("best_f1_threshold", 0.5)),
        metrics=pack.get("metrics", None)
    )


def score_winner_data(df: pd.DataFrame, model_pack: WinnerModelPack,
                     proba_col: str = "winner_proba") -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Score data using winner classifier model.

    Returns:
        - DataFrame with valid rows (after masking)
        - Probability scores
        - Boolean mask indicating which rows were valid
    """
    X, mask = prep_winner_like_training(
        df,
        model_pack.features,
        medians=model_pack.medians,
        impute_missing=model_pack.impute_missing
    )

    proba = model_pack.model.predict_proba(X)[:, 1]
    scored_df = df.loc[mask].copy()
    scored_df[proba_col] = proba

    return scored_df, proba, mask


def apply_winner_threshold(df: pd.DataFrame, proba_col: str, pred_col: str,
                          threshold: float) -> pd.DataFrame:
    """Apply threshold to probability scores to generate predictions."""
    df = df.copy()
    df[pred_col] = (df[proba_col] >= threshold).astype(int)
    return df


def select_winner_threshold(proba: np.ndarray, y: Optional[np.ndarray],
                           fixed_threshold: Optional[float] = None,
                           use_pack_f1: bool = True,
                           best_f1_threshold: float = 0.5,
                           auto_calibrate: bool = False,
                           target_precisions: Optional[List[float]] = None,
                           target_recalls: Optional[List[float]] = None) -> Tuple[float, Optional[pd.DataFrame]]:
    """
    Select appropriate threshold based on configuration.

    Returns:
        - Selected threshold
        - Optional threshold table (for auto-calibration)
    """
    if fixed_threshold is not None:
        return fixed_threshold, None

    if auto_calibrate and y is not None and (target_precisions or target_recalls):
        target_precisions = target_precisions or []
        target_recalls = target_recalls or []
        return pick_threshold_auto(y, proba, target_precisions, target_recalls)

    if use_pack_f1:
        return best_f1_threshold, None

    return 0.5, None


def calculate_winner_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Calculate standard classification metrics."""
    if len(np.unique(y_true)) <= 1:
        return {"auc_roc": float('nan'), "auc_prc": float('nan')}

    auc_roc = roc_auc_score(y_true, proba)
    auc_prc = average_precision_score(y_true, proba)

    return {"auc_roc": auc_roc, "auc_prc": auc_prc}


def extract_labels(df: pd.DataFrame, train_target: str = "return_mon",
                  train_epsilon: float = 0.0) -> Optional[np.ndarray]:
    """Extract binary labels from DataFrame based on target column."""
    if train_target in df.columns:
        return (pd.to_numeric(df[train_target], errors="coerce") > train_epsilon).astype(int).values
    elif "win" in df.columns:
        return df["win"].astype(int).values
    return None


def write_winner_summary(output_path: str, df: pd.DataFrame, threshold: float,
                        pred_col: str, metrics: Optional[Dict[str, float]] = None) -> None:
    """Write scoring summary to JSON file."""
    summary = {
        "rows_scored": int(len(df)),
        "threshold": float(threshold),
        "predicted_winners": int(df[pred_col].sum()),
        "coverage": float(df[pred_col].mean()),
    }

    if metrics:
        summary.update({f"metric_{k}": float(v) for k, v in metrics.items()})

    json_path = Path(output_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)


def write_winner_metrics(output_path: str, metrics: Dict[str, float]) -> None:
    """Write metrics to text file."""
    metrics_path = Path(output_path).with_suffix(".metrics.txt")
    with open(metrics_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key.upper().replace('_', '-')}: {value:.6f}\n")


def cleanup_columns_for_production(df: pd.DataFrame, columns_to_drop: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove columns that are not needed in production output."""
    default_drops = [
        'baseSymbolType', 'expirationDate', 'strike', 'moneyness',
        'breakEvenBid', 'percentToBreakEvenBid', 'tradeTime', 'symbol_norm',
        'impliedVolatilityRank1y', 'delta', 'breakEvenProbability',
        'expirationType', 'symbolType', 'entry_credit', 'exit_intrinsic',
        'total_pnl', 'return_pct', 'ret_2d', 'ret_5d', 'ret_2d_norm',
        'ret_5d_norm', 'prev_close', 'prev_close_minus_ul',
        'prev_close_minus_ul_pct', 'log1p_DTE', 'bid'
    ]

    # Add GEX columns
    gex_columns = [col for col in df.columns if col.startswith("gex_")]
    columns_to_drop = (columns_to_drop or []) + default_drops + gex_columns

    return df.drop(columns=columns_to_drop, errors='ignore')