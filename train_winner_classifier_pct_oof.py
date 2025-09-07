#!/usr/bin/env python3
"""
train_winner_classifier_pct_oof.py

Minimal-mod OOF version of your winner classifier trainer.
- Uses OOF CV instead of a single train/test split.
- If 'trade_date' column exists -> TimeSeriesSplit; else StratifiedKFold.
- Saves OOF predictions, OOF metrics, and trains a final model on ALL data for export.

Inputs via .env (same as original):
  WINNER_INPUT                          # CSV path (or OUTPUT_CSV if you prefer your enriched output)
  WINNER_OUTPUT_DIR
  WINNER_FEATURES                       # optional explicit features (comma or JSON)
  WINNER_ID_COLS                        # optional ID columns
  WINNER_RANDOM_STATE=42
  WINNER_CLASSIFIER_N_ESTIMATORS=400
  WINNER_CLASS_WEIGHT=balanced_subsample
  WINNER_MAX_DEPTH=
  WINNER_MIN_SAMPLES_LEAF=1
  WINNER_MIN_SAMPLES_SPLIT=2
  WINNER_IMPUTE_MISSING=1
  WINNER_USE_WEIGHTS=1
  WINNER_WEIGHT_ALPHA=0.02
  WINNER_WEIGHT_MIN=0.5
  WINNER_WEIGHT_MAX=10.0
  WINNER_TRAIN_TARGET=return_mon

New/optional:
  WINNER_OOF_FOLDS=5
  WINNER_TIME_SERIES=auto   # 'auto' (use TimeSeries if trade_date present), '1' to force TimeSeries, '0' to force Stratified
"""

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.utils import shuffle
from dotenv import load_dotenv
import joblib

# --- Helper imports: try service.* first, then fall back to local modules ---
try:
    from service.utils import BASE_FEATS, GEX_FEATS, NEW_FEATS
except Exception:
    # define light fallbacks if your utils don't expose these (or customize via WINNER_FEATURES)
    BASE_FEATS, GEX_FEATS, NEW_FEATS = [], [], []

try:
    from service.preprocess import add_dte_and_normalized_returns
except Exception:
    # fallback: no-op if you don't have this; replace with your own prep
    def add_dte_and_normalized_returns(df: pd.DataFrame) -> pd.DataFrame:
        return df

from sklearn.inspection import permutation_importance


# ---------- Small utility helpers ----------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _parse_list_env(val: str) -> List[float]:
    if val is None:
        return []
    s = val.strip()
    if not s:
        return []
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return [float(x) for x in arr]
    except Exception:
        pass
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _parse_str_list(val: str) -> List[str]:
    if val is None:
        return []
    s = val.strip()
    if not s:
        return []
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return [str(x) for x in arr]
    except Exception:
        pass
    return [x.strip() for x in s.split(",") if x.strip()]

def _maybe_none(val: str):
    if val is None or str(val).strip() == "":
        return None
    return val

def _maybe_int(val: str):
    return None if _maybe_none(val) is None else int(val)

def _maybe_float(val: str):
    return None if _maybe_none(val) is None else float(val)

def _parse_bool(val: str, default=False):
    if val is None:
        return default
    return str(val).strip().lower() in {"1","true","yes","y","on"}

def build_label(df: pd.DataFrame, target_col: str) -> pd.Series:
    if target_col not in df.columns:
        raise ValueError(f"Column `{target_col}` not found.")
    return (pd.to_numeric(df[target_col], errors="coerce") > 0).astype(int)

def select_features(df: pd.DataFrame, explicit: List[str], id_cols: List[str]) -> List[str]:
    if explicit:
        for c in explicit:
            if c not in df.columns:
                raise ValueError(f"Feature '{c}' not in dataframe.")
        return explicit
    exclude = set(id_cols or [])
    exclude.update({"return_pct","return_mon","return_ann"})
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if c not in exclude]
    if not feats:
        raise ValueError("No numeric features detected. Provide WINNER_FEATURES in .env")
    return feats

def compute_medians(df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    med = {}
    for c in features:
        med[c] = float(pd.to_numeric(df[c], errors="coerce").median())
    return med

def apply_impute(df: pd.DataFrame, features: List[str], medians: Dict[str, float]) -> pd.DataFrame:
    X = df[features].copy()
    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(medians[c])
    return X

def drop_na_rows(X: pd.DataFrame, y: pd.Series, w: pd.Series = None):
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    if w is not None:
        return X[mask], y[mask], w[mask]
    return X[mask], y[mask], None

def pick_threshold_by_target(y_true: np.ndarray, proba: np.ndarray,
                             targets_recall: List[float],
                             targets_precision: List[float]) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    def metrics_at(thr: float):
        yhat = (proba >= thr).astype(int)
        pr = precision_score(y_true, yhat, zero_division=0)
        rc = recall_score(y_true, yhat, zero_division=0)
        f1 = f1_score(y_true, yhat, zero_division=0)
        keep = float(yhat.mean())
        return pr, rc, f1, keep

    rows = []
    if targets_recall:
        for tgt in targets_recall:
            chosen_thr, chosen_m = None, None
            for thr in sorted(thresholds, reverse=True):
                pr, rc, f1, keep = metrics_at(thr)
                if rc >= tgt:
                    chosen_thr, chosen_m = thr, (pr, rc, f1, keep)
            if chosen_thr is None and len(thresholds):
                thr = float(min(thresholds))
                chosen_m = metrics_at(thr)
                chosen_thr = thr
            if chosen_thr is not None:
                pr, rc, f1, keep = chosen_m
                rows.append(dict(target_type="recall", target=tgt, threshold=chosen_thr,
                                 precision=pr, recall=rc, f1=f1, coverage=keep))
    if targets_precision:
        for tgt in targets_precision:
            chosen_thr, chosen_m = None, None
            for thr in sorted(thresholds):
                pr, rc, f1, keep = metrics_at(thr)
                if pr >= tgt:
                    chosen_thr, chosen_m = thr, (pr, rc, f1, keep)
                    break
            if chosen_thr is None and len(thresholds):
                thr = float(max(thresholds))
                chosen_m = metrics_at(thr)
                chosen_thr = thr
            if chosen_thr is not None:
                pr, rc, f1, keep = chosen_m
                rows.append(dict(target_type="precision", target=tgt, threshold=chosen_thr,
                                 precision=pr, recall=rc, f1=f1, coverage=keep))
    return pd.DataFrame(rows)

def save_feature_importances(clf, X_ref, y_ref, feats, outdir):
    out_csv = os.path.join(outdir, "winner_feature_importances.csv")
    out_png = os.path.join(outdir, "winner_feature_importances.png")

    fi_df = None
    if hasattr(clf, "feature_importances_"):
        fi = np.asarray(clf.feature_importances_, dtype=float)
        fi_df = pd.DataFrame({"feature": feats, "importance": fi}).sort_values("importance", ascending=False)

    if fi_df is None:
        try:
            perm = permutation_importance(clf, X_ref, y_ref, n_repeats=10, random_state=42, n_jobs=-1)
            fi_df = pd.DataFrame({"feature": feats, "importance": perm.importances_mean, "importance_std": perm.importances_std})\
                    .sort_values("importance", ascending=False)
        except Exception as e:
            print(f"[WARN] Permutation importance failed: {e}")
            return None

    fi_df.to_csv(out_csv, index=False)

    top = fi_df.head(30).iloc[::-1]
    plt.figure(figsize=(8, max(6, 0.3 * len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title("Winner Classifier — Feature Importance (final model)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return fi_df


def main():
    load_dotenv()

    # Required (support both WINNER_INPUT and OUTPUT_CSV, like your original)
    CSV = os.getenv("WINNER_INPUT") or os.getenv("OUTPUT_CSV")
    OUTDIR = os.getenv("WINNER_OUTPUT_DIR")
    MODEL_NAME = os.getenv("WINNER_MODEL_NAME", "winner_classifier_model.pkl")

    if not CSV or not OUTDIR:
        raise SystemExit("WINNER_INPUT (or OUTPUT_CSV) and WINNER_OUTPUT_DIR must be set in .env")

    ensure_dir(OUTDIR)

    # Optional / defaults
    ADDED_FEATS = ["is_earnings_week","is_earnings_window","post_earnings_within_3d"]

    # If you supply WINNER_FEATURES in .env, it overrides the default list constructed below.
    FEATURES_ENV = _parse_str_list(os.getenv("WINNER_FEATURES"))
    if FEATURES_ENV:
        FEATURES = FEATURES_ENV
    else:
        FEATURES = BASE_FEATS + NEW_FEATS + ["gex_neg","gex_center_abs_strike","gex_total_abs"] + ADDED_FEATS

    ID_COLS         = _parse_str_list(os.getenv("WINNER_ID_COLS"))
    RANDOM_STATE    = int(os.getenv("WINNER_RANDOM_STATE", "42"))
    N_ESTIMATORS    = int(os.getenv("WINNER_CLASSIFIER_N_ESTIMATORS", "400"))
    CLASS_WEIGHT    = _maybe_none(os.getenv("WINNER_CLASS_WEIGHT", "balanced_subsample"))
    MAX_DEPTH       = _maybe_int(os.getenv("WINNER_MAX_DEPTH", ""))
    MIN_S_LEAF      = int(os.getenv("WINNER_MIN_SAMPLES_LEAF", "1"))
    MIN_S_SPLIT     = int(os.getenv("WINNER_MIN_SAMPLES_SPLIT", "2"))

    IMPUTE_MISSING  = _parse_bool(os.getenv("WINNER_IMPUTE_MISSING", "1"))
    USE_WEIGHTS     = _parse_bool(os.getenv("WINNER_USE_WEIGHTS", "1"))
    WEIGHT_ALPHA    = float(os.getenv("WINNER_WEIGHT_ALPHA", "0.02"))
    WEIGHT_MIN      = float(os.getenv("WINNER_WEIGHT_MIN", "0.5"))
    WEIGHT_MAX      = float(os.getenv("WINNER_WEIGHT_MAX", "10.0"))
    TRAIN_TARGET    = os.getenv("WINNER_TRAIN_TARGET", "return_mon").strip()

    # OOF controls
    OOF_FOLDS       = int(os.getenv("WINNER_OOF_FOLDS", "5"))
    TIME_SERIES     = os.getenv("WINNER_TIME_SERIES", "auto").strip().lower()  # 'auto' / '1' / '0'

    TARGETS_RECALL    = _parse_list_env(os.getenv("WINNER_TARGET_RECALL", ""))
    TARGETS_PRECISION = _parse_list_env(os.getenv("WINNER_TARGET_PRECISION", ""))

    # Load
    df = pd.read_csv(CSV)
    df = add_dte_and_normalized_returns(df)
    if TRAIN_TARGET not in df.columns:
        raise ValueError(f"Training target '{TRAIN_TARGET}' not found in DataFrame")

    # Sort if time present
    has_time = 'trade_date' in df.columns
    if has_time:
        try:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
        except Exception:
            pass
        df = df.sort_values('trade_date').reset_index(drop=True)
    else:
        # keep original shuffle behavior when no time available
        df = shuffle(df, random_state=RANDOM_STATE)

    # Label
    y = build_label(df, TRAIN_TARGET).astype(int).values

    # Features (either explicit from .env or inferred numeric)
    feats = select_features(df, FEATURES, ID_COLS)

    # Weights (optional)
    wgt = None
    if USE_WEIGHTS:
        ret = pd.to_numeric(df[TRAIN_TARGET], errors="coerce").fillna(0.0)
        wgt = 1.0 + WEIGHT_ALPHA * ret.abs()
        wgt = np.clip(wgt, WEIGHT_MIN, WEIGHT_MAX).values

    # Medians & impute
    # Minimal-change approach: global medians used per-fold (OK for RF OOF scoring).
    medians_global = None
    if IMPUTE_MISSING:
        medians_global = compute_medians(df, feats)

    # Prepare CV
    if (TIME_SERIES == '1') or (TIME_SERIES == 'auto' and has_time):
        splitter = TimeSeriesSplit(n_splits=OOF_FOLDS)
        split_iter = splitter.split(df)
        split_kind = "TimeSeriesSplit"
    else:
        splitter = StratifiedKFold(n_splits=OOF_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(df, y)
        split_kind = "StratifiedKFold"

    # OOF
    proba_oof = np.full(len(df), np.nan, dtype=float)
    fold_idx  = np.full(len(df), -1, dtype=int)

    for k, (tr, va) in enumerate(split_iter):
        if IMPUTE_MISSING:
            medians_k = compute_medians(df.iloc[tr], feats)
            Xtr = apply_impute(df.iloc[tr], feats, medians_k)
            Xva = apply_impute(df.iloc[va], feats, medians_k)
        else:
            Xtr = df.iloc[tr][feats].apply(pd.to_numeric, errors="coerce")
            Xva = df.iloc[va][feats].apply(pd.to_numeric, errors="coerce")
            Xtr, ytr, wgtr = drop_na_rows(Xtr, pd.Series(y[tr]), None if wgt is None else pd.Series(wgt[tr]))
            # re-slice va to numeric
            mask_va = ~Xva.isna().any(axis=1)
            Xva = Xva[mask_va]; yva = y[va][mask_va.values]
            # adjust indices
            va = df.index[va][mask_va.values].to_numpy()
        ytr = y[tr] if IMPUTE_MISSING else ytr.values
        wgtr = None if wgt is None else (wgt[tr] if IMPUTE_MISSING else wgtr.values)

        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_S_LEAF,
            min_samples_split=MIN_S_SPLIT,
            n_jobs=-1,
            random_state=RANDOM_STATE + k,
            class_weight=CLASS_WEIGHT,
        )
        clf.fit(Xtr, ytr, sample_weight=wgtr)

        pva = clf.predict_proba(Xva)[:, 1]
        proba_oof[va] = pva
        fold_idx[va] = k

    # Any remaining NaNs -> fill with median prob
    nan_mask = np.isnan(proba_oof)
    if nan_mask.any():
        fill_val = np.nanmedian(proba_oof)
        proba_oof[nan_mask] = fill_val

    # OOF metrics
    roc_auc = roc_auc_score(y, proba_oof) if len(np.unique(y)) > 1 else float("nan")
    pr_auc  = average_precision_score(y, proba_oof)

    # PR vs coverage on OOF
    precision, recall, thresholds = precision_recall_curve(y, proba_oof)
    coverage = [ (proba_oof >= t).mean() for t in thresholds ]
    pr_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
        "coverage": coverage
    })
    pr_csv_path = os.path.join(OUTDIR, "precision_recall_coverage.csv")
    pr_df.to_csv(pr_csv_path, index=False)

    # Threshold table at requested targets (on OOF)
    thr_table = pick_threshold_by_target(y, proba_oof, TARGETS_RECALL, TARGETS_PRECISION)
    thr_csv_path = os.path.join(OUTDIR, "threshold_table.csv")
    thr_table.to_csv(thr_csv_path, index=False)

    # Save per-row OOF probabilities
    oof_out = pd.DataFrame({
        "row_idx": np.arange(len(df)),
        "proba_oof": proba_oof,
        "label": y,
        "fold": fold_idx
    })
    if ID_COLS:
        for c in ID_COLS:
            if c in df.columns:
                oof_out[c] = df[c].values
    oof_out.to_csv(os.path.join(OUTDIR, "winner_scores_oof.csv"), index=False)

    # Plot PR/coverage
    plt.figure(figsize=(8,6))
    plt.plot(pr_df["coverage"], pr_df["precision"], label="Precision vs Coverage (OOF)")
    plt.plot(pr_df["coverage"], pr_df["recall"], label="Recall vs Coverage (OOF)")
    plt.xlabel("Coverage (fraction predicted winners)")
    plt.ylabel("Score")
    plt.title(f"Precision & Recall vs Coverage — Winner Classifier OOF ({split_kind}, folds={OOF_FOLDS})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTDIR, "precision_recall_coverage.png"), dpi=150)
    plt.close()

    # Train FINAL model on ALL data for export
    if IMPUTE_MISSING:
        X_all = apply_impute(df, feats, medians_global if medians_global is not None else compute_medians(df, feats))
    else:
        X_all = df[feats].apply(pd.to_numeric, errors="coerce")
        X_all, y_all, _ = drop_na_rows(X_all, pd.Series(y), None)
        y = y_all.values  # replace with filtered y if needed

    clf_final = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_S_LEAF,
        min_samples_split=MIN_S_SPLIT,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight=CLASS_WEIGHT,
    )
    clf_final.fit(X_all, y)

    # Save feature importances (using final model & X_all)
    try:
        _ = save_feature_importances(clf_final, X_all, y, feats, OUTDIR)
    except Exception as e:
        print(f"[WARN] feature importance failed: {e}")

    # Best-F1 threshold (OOF)
    best_f1, best_thr = -1.0, 0.5
    for thr in thresholds:
        yhat = (proba_oof >= thr).astype(int)
        f1 = f1_score(y, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    cm = confusion_matrix(y, (proba_oof >= best_thr).astype(int))
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "roc_auc_oof": float(roc_auc),
        "pr_auc_oof": float(pr_auc),
        "best_f1_threshold_oof": float(best_thr),
        "best_f1_oof": float(best_f1),
        "coverage_at_best_f1_oof": float((proba_oof >= best_thr).mean()),
        "confusion_at_best_f1_oof": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "features": feats,
        "impute_missing": bool(IMPUTE_MISSING),
        "use_weights": bool(USE_WEIGHTS),
        "cv": {"kind": split_kind, "folds": int(OOF_FOLDS)}
    }

    with open(os.path.join(OUTDIR, "winner_classifier_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model pack
    pack = {
        "model": clf_final,
        "features": feats,
        "medians": (compute_medians(df, feats) if IMPUTE_MISSING else None),
        "impute_missing": bool(IMPUTE_MISSING),
        "targets_table_path": thr_csv_path,
        "metrics": metrics,
        "label": f"{TRAIN_TARGET} > 0",
        "oof_scores_path": os.path.join(OUTDIR, "winner_scores_oof.csv"),
    }
    joblib.dump(pack, os.path.join(OUTDIR, MODEL_NAME))

    print(f"✅ Winner classifier trained with OOF. ROC AUC(O): {roc_auc:.4f}, PR AUC(O): {pr_auc:.4f}")
    print(f"CV: {split_kind}, folds={OOF_FOLDS}")
    print(f"Outputs saved in: {OUTDIR}")
    print(f"- winner_scores_oof.csv  (per-row OOF proba + fold)")
    print(f"- precision_recall_coverage.csv / .png  (OOF)")
    print(f"- threshold_table.csv    (OOF targets)")
    print(f"- winner_classifier_metrics.json  (OOF metrics)")
    print(f"- {MODEL_NAME} (final model pack)")

if __name__ == "__main__":
    main()
