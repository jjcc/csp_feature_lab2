#!/usr/bin/env python3
"""
train_winner_classifier_env.py

Winner Classifier trainer fully driven by .env (dotenv).
New label: winner := (return_pct > 0).

Outputs:
- model_pack.pkl            (model, features, medians if used, thresholds table, metrics)
- precision_recall_coverage.csv
- threshold_table.csv       (per-target thresholds & metrics if targets provided)
- ml_classifier_metrics.json
- precision_recall_coverage.png

Key behaviors:
- Features: use WINNER_FEATURES (comma/JSON) or auto-detect numeric columns (excluding id cols and return_pct)
- Missing values: by default, median-impute on the TRAIN split only (set WINNER_IMPUTE_MISSING=0 to drop rows instead)
- Weighting: optional sample weights from return_pct magnitude (WINNER_USE_WEIGHTS=1)
- Threshold picking for reporting: from WINNER_TARGET_RECALL and/or WINNER_TARGET_PRECISION lists

.env variables (required):
  WINNER_INPUT=path/to/labeled_trades.csv
  WINNER_OUTPUT_DIR=./output_winner

.env variables (optional):
  WINNER_FEATURES=feat1,feat2,[...]      # comma or JSON list
  WINNER_ID_COLS=ticker,entry_date       # comma or JSON list
  WINNER_TEST_SIZE=0.2
  WINNER_RANDOM_STATE=42
  WINNER_CLASSIFIER_N_ESTIMATORS=400
  WINNER_CLASS_WEIGHT=balanced_subsample # or None
  WINNER_MAX_DEPTH=                      # empty -> None
  WINNER_MIN_SAMPLES_LEAF=1
  WINNER_MIN_SAMPLES_SPLIT=2

  # Imputation vs row-drop for NaNs
  WINNER_IMPUTE_MISSING=1                # 1=median impute (train medians), 0=drop rows with NaNs

  # Optional weighting tied to |return_pct|
  WINNER_USE_WEIGHTS=1
  WINNER_WEIGHT_ALPHA=0.02               # weight = 1 + alpha * |return_pct|, clipped between [WINNER_WEIGHT_MIN, WINNER_WEIGHT_MAX]
  WINNER_WEIGHT_MIN=0.5
  WINNER_WEIGHT_MAX=10.0

  # Threshold targets (either can be a single float or comma/JSON list)
  WINNER_TARGET_RECALL=0.65,0.83,0.90
  WINNER_TARGET_PRECISION=                # e.g. 0.55 or 0.50,0.60

"""

import json
import os
from typing import Dict, List, Tuple

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
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from dotenv import load_dotenv
import joblib

from service.utils import BASE_FEATS, GEX_FEATS, NEW_FEATS
from service.preprocess import add_dte_and_normalized_returns
from sklearn.inspection import permutation_importance


# ---------- Helpers ----------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _parse_list_env(val: str) -> List[float]:
    """Return list of floats parsed from JSON or comma-separated string. Empty -> []"""
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
    # comma-separated
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


# ---------- Core ----------

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
    exclude.add("return_pct","return_mon","return_ann")
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
    """Build a table of thresholds and metrics at requested recall/precision targets."""
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
        # choose highest threshold that still achieves >= target recall
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
        # choose lowest threshold that achieves >= target precision
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

def save_feature_importances(clf, X_test, y_test, feats, outdir):
    """Save feature importances (model-based if available; else permutation). Also plot a bar chart."""
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    out_csv = os.path.join(outdir, "winner_feature_importances.csv")
    out_png = os.path.join(outdir, "winner_feature_importances.png")

    fi_df = None

    # 1) Model-native importances (fast)
    if hasattr(clf, "feature_importances_"):
        fi = np.asarray(clf.feature_importances_, dtype=float)
        fi_df = pd.DataFrame({"feature": feats, "importance": fi})
        fi_df.sort_values("importance", ascending=False, inplace=True)

    # 2) Fallback: permutation importance (robust, slower)
    if fi_df is None:
        try:
            perm = permutation_importance(
                clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
            )
            fi_df = pd.DataFrame({"feature": feats, "importance": perm.importances_mean})
            fi_df["importance_std"] = perm.importances_std
            fi_df.sort_values("importance", ascending=False, inplace=True)
        except Exception as e:
            print(f"[WARN] Permutation importance failed: {e}")
            return None

    # Save CSV
    fi_df.to_csv(out_csv, index=False)

    # Plot (top 30 for readability)
    top = fi_df.head(30).iloc[::-1]  # reverse for horizontal bar order
    plt.figure(figsize=(8, max(6, 0.3 * len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title("Winner Classifier — Feature Importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    return fi_df


def main():
    load_dotenv()

    # Required
    #CSV = os.getenv("WINNER_INPUT")
    CSV= os.getenv("OUTPUT_CSV") # from enriched
    OUTDIR = os.getenv("WINNER_OUTPUT_DIR")
    MODEL_NAME = os.getenv("WINNER_MODEL_NAME", "winner_classifier_model.pkl")

    if not CSV or not OUTDIR:
        raise SystemExit("WINNER_INPUT and WINNER_OUTPUT_DIR must be set in .env")

    ensure_dir(OUTDIR)

    # Optional
    #FEATURES        = _parse_str_list(os.getenv("WINNER_FEATURES"))
    #FEATURES        = BASE_FEATS + GEX_FEATS + NEW_FEATS
    ADDED_FEATS = [
        "is_earnings_week","is_earnings_window","post_earnings_within_3d",
        ]
    FEATURES        = BASE_FEATS + NEW_FEATS + ["gex_neg","gex_center_abs_strike","gex_total_abs"] + ADDED_FEATS

    ID_COLS         = _parse_str_list(os.getenv("WINNER_ID_COLS"))
    TEST_SIZE       = float(os.getenv("WINNER_TEST_SIZE", "0.2"))
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

    TARGETS_RECALL    = _parse_list_env(os.getenv("WINNER_TARGET_RECALL", ""))
    TARGETS_PRECISION = _parse_list_env(os.getenv("WINNER_TARGET_PRECISION", ""))

    # Load
    df = pd.read_csv(CSV)
    df = add_dte_and_normalized_returns(df)
    if TRAIN_TARGET not in df.columns:
        raise ValueError(f"Training target '{TRAIN_TARGET}' not found in DataFrame")
    df = shuffle(df, random_state=RANDOM_STATE)

    # Label
    y = build_label(df, TRAIN_TARGET)

    # Features
    feats = select_features(df, FEATURES, ID_COLS)

    # Weights (optional)
    wgt = None
    if USE_WEIGHTS:
        #ret = pd.to_numeric(df["return_pct"], errors="coerce").fillna(0.0)
        ret = pd.to_numeric(df[TRAIN_TARGET], errors="coerce").fillna(0.0)
        wgt = 1.0 + WEIGHT_ALPHA * ret.abs()
        wgt = np.clip(wgt, WEIGHT_MIN, WEIGHT_MAX)

    # Split
    stratify = y if y.nunique() > 1 else None
    df_train, df_test, y_train, y_test, wgt_train, wgt_test, idx_train, idx_test = train_test_split(
        df, y, wgt, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    #X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    #    X, y, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    #)


    # Prepare X with either imputation or row-drop
    medians = None
    if IMPUTE_MISSING:
        medians = compute_medians(df_train, feats)
        X_train = apply_impute(df_train, feats, medians)
        X_test = apply_impute(df_test, feats, medians)
    else:
        X_train = df_train[feats].apply(pd.to_numeric, errors="coerce")
        X_test = df_test[feats].apply(pd.to_numeric, errors="coerce")
        X_train, y_train, wgt_train = drop_na_rows(X_train, y_train, wgt_train)
        X_test, y_test, wgt_test = drop_na_rows(X_test, y_test, wgt_test)

    # Train
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_S_LEAF,
        min_samples_split=MIN_S_SPLIT,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight=CLASS_WEIGHT,
    )
    clf.fit(X_train, y_train, sample_weight=wgt_train if (USE_WEIGHTS and wgt_train is not None) else None)

    # Evaluate
    y_proba_test = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba_test)
    pr_auc = average_precision_score(y_test, y_proba_test)

    # Precision-Recall vs Coverage CSV + plot
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba_test)
    coverage = [ (y_proba_test >= t).mean() for t in thresholds ]
    pr_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
        "coverage": coverage
    })
    pr_csv_path = os.path.join(OUTDIR, "precision_recall_coverage.csv")
    pr_df.to_csv(pr_csv_path, index=False)

    # === NEW: Save feature importances ===
    fi_df = save_feature_importances(clf, X_test, y_test, feats, OUTDIR)
    fi_path = os.path.join(OUTDIR, "winner_feature_importances.csv")
    fi_df.to_csv(fi_path, index=False)

    # save data with lable of test or not
   # === NEW: Save per-row probabilities with split labels ===
    try:
        label_series = y.astype(int) 
        y_proba_train = clf.predict_proba(X_train)[:, 1]
        df_out = pd.DataFrame({
            "row_idx": df.index,
            "proba": np.nan,
            "label": label_series,
        })
        df_out.set_index("row_idx", inplace=True)  # ← critical
    
        df_out.loc[idx_train, "proba"] = y_proba_train
        df_out.loc[idx_train, "is_train"] = 1
        df_out.loc[idx_test,  "proba"] = y_proba_test
        df_out.loc[idx_test,  "is_train"] = 0

        if ID_COLS:
            df_out[ID_COLS] = df.loc[df_out.index, ID_COLS]
    
        # write with row_idx as a column
        df_out.reset_index().to_csv(os.path.join(OUTDIR, "winner_scores_split.csv"), index=False)
    
        # optional sanity checks
        # assert int((df_out["is_train"]==1).sum()) == len(idx_train)
        # assert int((df_out["is_train"]==0).sum()) == len(idx_test)
    
    except Exception as e:
        print(f"[WARN] Could not save split scores: {e}")
 

    # plot
    plt.figure(figsize=(8,6))
    plt.plot(pr_df["coverage"], pr_df["precision"], label="Precision vs Coverage")
    plt.plot(pr_df["coverage"], pr_df["recall"], label="Recall vs Coverage")
    plt.xlabel("Coverage (fraction predicted winners)")
    plt.ylabel("Score")
    plt.title("Precision & Recall vs Coverage — Winner Classifier (return_pct > 0)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTDIR, "precision_recall_coverage.png"), dpi=150)
    plt.close()

    # Threshold table for requested targets
    thr_table = pick_threshold_by_target(y_test.values, y_proba_test, TARGETS_RECALL, TARGETS_PRECISION)
    thr_csv_path = os.path.join(OUTDIR, "threshold_table.csv")
    thr_table.to_csv(thr_csv_path, index=False)

    # A simple "balanced F1 best" line (optional info for metrics JSON)
    best_f1, best_thr = -1.0, 0.5
    for thr in thresholds:
        yhat = (y_proba_test >= thr).astype(int)
        f1 = f1_score(y_test, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    # Confusion at best F1
    yhat_best = (y_proba_test >= best_thr).astype(int)
    cm = confusion_matrix(y_test, yhat_best)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "best_f1_threshold": float(best_thr),
        "best_f1": float(best_f1),
        "coverage_at_best_f1": float(yhat_best.mean()),
        "confusion_at_best_f1": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "features": feats,
        "impute_missing": bool(IMPUTE_MISSING),
        "use_weights": bool(USE_WEIGHTS),
        "targets": {
            "recall": TARGETS_RECALL,
            "precision": TARGETS_PRECISION
        }
    }

    with open(os.path.join(OUTDIR, "winner_classifier_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model pack (includes medians + thresholds table path)
    pack = {
        "model": clf,
        "features": feats,
        "medians": medians,
        "impute_missing": bool(IMPUTE_MISSING),
        "targets_table_path": thr_csv_path,
        "metrics": metrics,
        "label": "return_pct > 0"
    }
    joblib.dump(pack, os.path.join(OUTDIR, MODEL_NAME))

    print(f"✅ Winner classifier trained. ROC AUC={roc_auc:.4f}, PR AUC={pr_auc:.4f}")
    print(f"Outputs saved in: {OUTDIR}")
    print(f"- precision_recall_coverage.csv")
    print(f"- precision_recall_coverage.png")
    print(f"- threshold_table.csv (requested targets)")
    print(f"- winner_classifier_metrics.json")
    print(f"- {MODEL_NAME}")

if __name__ == "__main__":
    main()
