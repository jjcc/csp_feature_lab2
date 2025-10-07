#!/usr/bin/env python3
"""
train_winner_classifier_pct_oof.py

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



Minimal-mod OOF version of winner classifier trainer.
- Uses OOF CV instead of a single train/test split.
- If 'trade_date' column exists -> TimeSeriesSplit; else StratifiedKFold.
- Saves OOF predictions, OOF metrics, and trains a final model on ALL data for export.

Inputs via .env (same as original):
  WINNER_INPUT                          # CSV path (or OUTPUT_CSV if prefer enriched output)
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
from service.utils import BASE_FEATS, GEX_FEATS, NEW_FEATS
from service.preprocess import add_dte_and_normalized_returns
from sklearn.inspection import permutation_importance
from service.env_config import getenv


# ---------- Configuration ----------

class WinnerClassifierConfig:
    """Configuration parser for Winner Classifier training."""

    def __init__(self):
        self._parse_config()

    def _parse_list_env(self, val: str) -> List[float]:
        """Return list of floats parsed from JSON or comma-separated string."""
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

    def _parse_str_list(self, val: str) -> List[str]:
        """Return list of strings parsed from JSON or comma-separated string."""
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

    def _maybe_none(self, val: str):
        """Return None if value is empty string."""
        if val is None or str(val).strip() == "":
            return None
        return val

    def _maybe_int(self, val: str):
        """Parse integer or return None if empty."""
        return None if self._maybe_none(val) is None else int(val)

    def _maybe_float(self, val: str):
        """Parse float or return None if empty."""
        return None if self._maybe_none(val) is None else float(val)

    def _parse_bool(self, val: str, default=False):
        """Parse boolean from string."""
        if val is None:
            return default
        return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _parse_config(self):
        """Parse all configuration from environment variables."""
        # I/O paths
        self.output_dir = getenv("WINNER_OUTPUT_DIR")
        self.input_csv = getenv("COMMON_OUTPUT_DIR") + "/" + getenv("COMMON_OUTPUT_CSV")
        self.model_name = getenv("WINNER_MODEL_NAME", "winner_classifier_model.pkl")

        # Features
        earning_feats = ["is_earnings_week", "is_earnings_window", "post_earnings_within_3d"]
        self.features = BASE_FEATS + NEW_FEATS + ["gex_neg", "gex_center_abs_strike", "gex_total_abs"]
        self.id_cols = self._parse_str_list(getenv("WINNER_ID_COLS"))

        # Model parameters
        self.random_state = int(getenv("WINNER_RANDOM_STATE", "42"))
        self.n_estimators = int(getenv("WINNER_CLASSIFIER_N_ESTIMATORS", "400"))
        self.class_weight = self._maybe_none(getenv("WINNER_CLASS_WEIGHT", "balanced_subsample"))
        self.max_depth = self._maybe_int(getenv("WINNER_MAX_DEPTH", ""))
        self.min_samples_leaf = int(getenv("WINNER_MIN_SAMPLES_LEAF", "1"))
        self.min_samples_split = int(getenv("WINNER_MIN_SAMPLES_SPLIT", "2"))
        self.model_type = getenv("WINNER_MODEL_TYPE", "rf").lower()

        # Preprocessing
        self.impute_missing = self._parse_bool(getenv("WINNER_IMPUTE_MISSING", "1"))
        self.use_weights = self._parse_bool(getenv("WINNER_USE_WEIGHTS", "1"))
        self.weight_alpha = float(getenv("WINNER_WEIGHT_ALPHA", "0.02"))
        self.weight_min = float(getenv("WINNER_WEIGHT_MIN", "0.5"))
        self.weight_max = float(getenv("WINNER_WEIGHT_MAX", "10.0"))

        # Training
        self.train_target = getenv("WINNER_TRAIN_TARGET", "return_mon").strip()
        self.oof_folds = int(getenv("WINNER_OOF_FOLDS", "5"))
        self.time_series = getenv("WINNER_TIME_SERIES", "auto").strip().lower()

        # Evaluation
        self.targets_recall = self._parse_list_env(getenv("WINNER_TARGET_RECALL", ""))
        self.targets_precision = self._parse_list_env(getenv("WINNER_TARGET_PRECISION", ""))

        # Early stopping (for gradient boosting models)
        self.early_stopping_rounds = int(getenv("WINNER_EARLY_STOPPING_ROUNDS", "100"))
        self.valid_fraction = float(getenv("WINNER_VALID_FRACTION", "0.1"))

        # Validation
        if not self.input_csv or not self.output_dir:
            raise SystemExit("WINNER_INPUT (or OUTPUT_CSV) and WINNER_OUTPUT_DIR must be set in .env")


# ---------- Helpers ----------

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def select_features(df: pd.DataFrame, explicit: List[str], id_cols: List[str]) -> List[str]:
    """Select features either from explicit list or auto-detect numeric columns."""
    if explicit:
        for c in explicit:
            if c not in df.columns:
                raise ValueError(f"Feature '{c}' not in dataframe.")
        return explicit
    exclude = set(id_cols or [])
    exclude.update(["return_pct", "return_mon", "return_ann"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if c not in exclude]
    if not feats:
        raise ValueError("No numeric features detected. Provide WINNER_FEATURES in .env")
    return feats


# ---------- Data Preprocessing ----------

class DataPreprocessor:
    """Handles data preprocessing including feature selection, imputation, and weighting."""

    def __init__(self, config: WinnerClassifierConfig):
        self.config = config
        self.medians_global = None

    def compute_medians(self, df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
        """Compute median values for features for imputation."""
        medians = {}
        for c in features:
            medians[c] = float(pd.to_numeric(df[c], errors="coerce").median())
        return medians

    def apply_impute(self, df: pd.DataFrame, features: List[str], medians: Dict[str, float]) -> pd.DataFrame:
        """Apply median imputation to features."""
        X = df[features].copy()
        for c in features:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(medians[c])
        return X

    def drop_na_rows(self, X: pd.DataFrame, y: pd.Series, w: pd.Series = None):
        """Drop rows with NaN values."""
        mask = (~X.isna().any(axis=1)) & (~y.isna())
        if w is not None:
            return X[mask], y[mask], w[mask]
        return X[mask], y[mask], None

    def compute_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Compute sample weights based on return magnitude."""
        if not self.config.use_weights:
            return None
        ret = pd.to_numeric(df[self.config.train_target], errors="coerce").fillna(0.0)
        weights = 1.0 + self.config.weight_alpha * ret.abs()
        weights = np.clip(weights, self.config.weight_min, self.config.weight_max)
        return weights

    def prepare_data(self, df: pd.DataFrame):
        """Main data preparation pipeline."""
        # Add preprocessing if needed
        df = add_dte_and_normalized_returns(df)

        # Shuffle data
        df = df.sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)

        # Build binary label
        y = build_label(df, self.config.train_target)

        # Select features
        features = select_features(df, self.config.features, self.config.id_cols)

        # Compute sample weights
        weights = self.compute_sample_weights(df)

        # Check if we have time series data
        has_time = "trade_date" in df.columns

        # Store global medians if using imputation
        if self.config.impute_missing:
            self.medians_global = self.compute_medians(df, features)

        return df, y, features, weights, has_time


# ---------- Model Factory ----------

class ModelFactory:
    """Factory class for creating different types of classifiers."""

    @staticmethod
    def create_model(config: WinnerClassifierConfig, seed_offset: int = 0):
        """Create a classifier based on configuration."""
        seed = config.random_state + seed_offset

        if config.model_type == "lgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=int(os.getenv("LGBM_N_ESTIMATORS", "2000")),
                learning_rate=float(os.getenv("LGBM_LR", "0.05")),
                num_leaves=int(os.getenv("LGBM_NUM_LEAVES", "63")),
                max_depth=int(os.getenv("LGBM_MAX_DEPTH", "-1")),
                min_child_samples=int(os.getenv("LGBM_MIN_CHILD", "80")),
                subsample=float(os.getenv("LGBM_BAGGING_FRACTION", "1.0")),
                subsample_freq=int(os.getenv("LGBM_BAGGING_FREQ", "0")),
                colsample_bytree=float(os.getenv("LGBM_FEATURE_FRACTION", "0.8")),
                reg_lambda=float(os.getenv("LGBM_L2", "5.0")),
                objective="binary",
                random_state=seed,
                n_jobs=-1,
            )
        elif config.model_type == "catboost":
            from catboost import CatBoostClassifier
            # TODD: environment variables still use the old version
            return CatBoostClassifier(
                iterations=int(os.getenv("CAT_ITERS", "4000")),
                learning_rate=float(os.getenv("CAT_LR", "0.05")),
                depth=int(os.getenv("CAT_DEPTH", "6")),
                l2_leaf_reg=float(os.getenv("CAT_L2", "6.0")),
                loss_function="Logloss",
                eval_metric=os.getenv("CAT_EVAL_METRIC", "AUC"),
                random_seed=seed,
                verbose=False,
                task_type=os.getenv("CAT_TASK_TYPE", "CPU"),
            )
        else:  # Random Forest
            return RandomForestClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_samples_leaf=config.min_samples_leaf,
                min_samples_split=config.min_samples_split,
                n_jobs=-1,
                random_state=seed,
                class_weight=config.class_weight,
            )


# ---------- Core ----------

def build_label(df: pd.DataFrame, target_col: str, epsilon:float = 0.00) -> pd.Series:
    '''
    Build binary label series from target_col by thresholding at epsilon.
    '''
    if target_col not in df.columns:
        raise ValueError(f"Column `{target_col}` not found.")
    return (pd.to_numeric(df[target_col], errors="coerce") > epsilon).astype(int)


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
            # bottle neck: choose 1 in every 100
            count = 0
            for thr in sorted(thresholds):
                if count % 100 != 0:
                    count += 1
                    continue
                pr, rc, f1, keep = metrics_at(thr)
                if pr >= tgt:
                    chosen_thr, chosen_m = thr, (pr, rc, f1, keep)
                    break
                count += 1
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

    # 1) Model-native importances (fast)
    if hasattr(clf, "feature_importances_"):
        fi = np.asarray(clf.feature_importances_, dtype=float)
        fi_df = pd.DataFrame({"feature": feats, "importance": fi}).sort_values("importance", ascending=False)

    # 2) Fallback: permutation importance (robust, slower)
    if fi_df is None:
        try:
            perm = permutation_importance(
                clf, X_ref, y_ref, n_repeats=10, random_state=42, n_jobs=-1
            )
            fi_df = pd.DataFrame({"feature": feats, "importance": perm.importances_mean, "importance_std": perm.importances_std})\
                    .sort_values("importance", ascending=False)
        except Exception as e:
            print(f"[WARN] Permutation importance failed: {e}")
            return None

    # Save CSV
    fi_df.to_csv(out_csv, index=False)

    # Plot (top 30 for readability)
    top = fi_df.head(30).iloc[::-1]
    plt.figure(figsize=(8, max(6, 0.3 * len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title("Winner Classifier — Feature Importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    return fi_df




# ---------- Cross-Validation ----------

class CrossValidator:
    """Handles cross-validation logic and out-of-fold predictions."""

    def __init__(self, config: WinnerClassifierConfig, preprocessor: DataPreprocessor):
        self.config = config
        self.preprocessor = preprocessor

    def get_cv_splitter(self, df: pd.DataFrame, y: np.ndarray, has_time: bool):
        """Get appropriate cross-validation splitter."""
        if (self.config.time_series == '1') or (self.config.time_series == 'auto' and has_time):
            splitter = TimeSeriesSplit(n_splits=self.config.oof_folds)
            return splitter, splitter.split(df), "TimeSeriesSplit"
        else:
            splitter = StratifiedKFold(n_splits=self.config.oof_folds, shuffle=True, random_state=self.config.random_state)
            return splitter, splitter.split(df, y), "StratifiedKFold"

    def fit_fold_model(self, clf, Xtr, ytr, wgtr, Xva, yva, fold_k):
        """Fit model for a single fold with optional early stopping."""
        use_es = (self.config.model_type in {"lgbm", "catboost"} and
                  self.config.valid_fraction > 0 and len(Xva) > 0)

        if use_es:
            cut = int(len(Xtr) * (1.0 - self.config.valid_fraction))
            Xtr_fit, ytr_fit = Xtr.iloc[:cut], ytr[:cut]
            Xeval, yeval = Xtr.iloc[cut:], ytr[cut:]

            if self.config.model_type == "catboost":
                clf.fit(
                    Xtr_fit, ytr_fit,
                    sample_weight=(wgtr[:cut] if wgtr is not None else None),
                    eval_set=(Xeval, yeval),
                    use_best_model=True,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose=False
                )
            else:  # lgbm
                clf.fit(
                    Xtr_fit, ytr_fit,
                    sample_weight=wgtr[:cut] if wgtr is not None else None,
                    eval_set=[(Xeval, yeval)],
                    eval_metric="aucpr",
                    callbacks=[__import__("lightgbm").early_stopping(self.config.early_stopping_rounds, verbose=False)]
                )
        else:
            clf.fit(Xtr, ytr, sample_weight=wgtr)

        return clf

    def run_oof_cv(self, df: pd.DataFrame, y: np.ndarray, features: List[str],
                   weights: np.ndarray, has_time: bool):
        """Run out-of-fold cross-validation."""
        splitter, split_iter, split_kind = self.get_cv_splitter(df, y, has_time)

        proba_oof = np.full(len(df), np.nan, dtype=float)
        fold_idx = np.full(len(df), -1, dtype=int)

        for k, (tr, va) in enumerate(split_iter):
            # Prepare fold data
            if self.config.impute_missing:
                medians_k = self.preprocessor.compute_medians(df.iloc[tr], features)
                Xtr = self.preprocessor.apply_impute(df.iloc[tr], features, medians_k)
                Xva = self.preprocessor.apply_impute(df.iloc[va], features, medians_k)
            else:
                Xtr = df.iloc[tr][features].apply(pd.to_numeric, errors="coerce")
                Xva = df.iloc[va][features].apply(pd.to_numeric, errors="coerce")
                Xtr, ytr, wgtr = self.preprocessor.drop_na_rows(
                    Xtr, pd.Series(y[tr]),
                    None if weights is None else pd.Series(weights[tr])
                )
                mask_va = ~Xva.isna().any(axis=1)
                Xva = Xva[mask_va]
                va = df.index[va][mask_va.values].to_numpy()

            ytr = y[tr] if self.config.impute_missing else ytr.values
            wgtr = None if weights is None else (weights[tr] if self.config.impute_missing else wgtr.values)

            # Train model
            clf = ModelFactory.create_model(self.config, k)
            clf = self.fit_fold_model(clf, Xtr, ytr, wgtr, Xva, y[va], k)

            # Predict
            pva = clf.predict_proba(Xva)[:, 1]
            proba_oof[va] = pva
            fold_idx[va] = k

        # Fill any remaining NaNs
        nan_mask = np.isnan(proba_oof)
        if nan_mask.any():
            fill_val = np.nanmedian(proba_oof)
            proba_oof[nan_mask] = fill_val

        return proba_oof, fold_idx, split_kind


# ---------- Main Function ----------

def main():
    """Main training function with refactored structure."""
    # Initialize components
    config = WinnerClassifierConfig()
    ensure_dir(config.output_dir)

    preprocessor = DataPreprocessor(config)
    cv_handler = CrossValidator(config, preprocessor)

    # Load and prepare data
    df = pd.read_csv(config.input_csv)
    # alternative
    #input_csv1 = "output/labeled_trades_tr_t1_merged.csv"
    #input_csv2 = "output/labeled_trades_tr_t1_merged_minus.csv"
    input_csv3 = "output/labeled_trades_tr_A_B_merged.csv"

    #df = pd.read_csv(input_csv1)
    #df = pd.read_csv(input_csv2)
    df = pd.read_csv(input_csv3)

    df, y, features, weights, has_time = preprocessor.prepare_data(df)

    if config.train_target not in df.columns:
        raise ValueError(f"Training target '{config.train_target}' not found in DataFrame")

    # Timing
    start_time = pd.Timestamp.now()

    # Run OOF Cross-validation
    proba_oof, fold_idx, split_kind = cv_handler.run_oof_cv(df, y, features, weights, has_time)

    # Calculate OOF metrics
    roc_auc = roc_auc_score(y, proba_oof) if len(np.unique(y)) > 1 else float("nan")
    pr_auc = average_precision_score(y, proba_oof)

    cv_time = pd.Timestamp.now() - start_time
    print(f"Completed OOF scoring ({len(df)} rows, {config.oof_folds} folds, {split_kind}) in {cv_time}")
    print(f"OOF AUC-ROC={roc_auc:.4f}; AUC-PR={pr_auc:.4f}")

    # Generate evaluation outputs
    _save_evaluation_results(config, df, y, proba_oof, fold_idx, split_kind)

    # Train final model on all data
    final_model, final_train_time = _train_final_model(config, preprocessor, df, y, features)

    # Save final results
    _save_final_results(config, final_model, features, df, proba_oof, y, roc_auc, pr_auc, split_kind)

    print(f"✅ Winner classifier trained with OOF. ROC AUC(O): {roc_auc:.4f}, PR AUC(O): {pr_auc:.4f}")
    print(f"CV: {split_kind}, folds={config.oof_folds}")
    print(f"Outputs saved in: {config.output_dir}")


def _save_evaluation_results(config: WinnerClassifierConfig, df: pd.DataFrame, y: np.ndarray,
                           proba_oof: np.ndarray, fold_idx: np.ndarray, split_kind: str):
    """Save precision-recall curves, threshold tables, and OOF predictions."""
    precision, recall, thresholds = precision_recall_curve(y, proba_oof)
    coverage = [(proba_oof >= t).mean() for t in thresholds]

    # Save PR curve data
    pr_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
        "coverage": coverage
    })
    pr_csv_path = os.path.join(config.output_dir, "precision_recall_coverage.csv")
    pr_df.to_csv(pr_csv_path, index=False)

    # Threshold table at requested targets
    thr_table = pick_threshold_by_target(y, proba_oof, config.targets_recall, config.targets_precision)
    thr_csv_path = os.path.join(config.output_dir, "threshold_table.csv")
    thr_table.to_csv(thr_csv_path, index=False)

    # Save OOF predictions
    oof_out = pd.DataFrame({
        "row_idx": np.arange(len(df)),
        "proba_oof": proba_oof,
        "label": y,
        "fold": fold_idx
    })
    if config.id_cols:
        for c in config.id_cols:
            if c in df.columns:
                oof_out[c] = df[c].values
    oof_out.to_csv(os.path.join(config.output_dir, "winner_scores_oof.csv"), index=False)

    # Plot PR/coverage
    plt.figure(figsize=(8, 6))
    plt.plot(pr_df["coverage"], pr_df["precision"], label="Precision vs Coverage (OOF)")
    plt.plot(pr_df["coverage"], pr_df["recall"], label="Recall vs Coverage (OOF)")
    plt.xlabel("Coverage (fraction predicted winners)")
    plt.ylabel("Score")
    plt.title(f"Precision & Recall vs Coverage — Winner Classifier OOF ({split_kind}, folds={config.oof_folds})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.output_dir, "precision_recall_coverage.png"), dpi=150)
    plt.close()


def _train_final_model(config: WinnerClassifierConfig, preprocessor: DataPreprocessor,
                      df: pd.DataFrame, y: np.ndarray, features: List[str]):
    """Train final model on all available data."""
    start_time = pd.Timestamp.now()

    if config.impute_missing:
        medians = preprocessor.medians_global or preprocessor.compute_medians(df, features)
        X_all = preprocessor.apply_impute(df, features, medians)
    else:
        X_all = df[features].apply(pd.to_numeric, errors="coerce")
        X_all, y_filtered, _ = preprocessor.drop_na_rows(X_all, pd.Series(y), None)
        y = y_filtered.values

    clf_final = ModelFactory.create_model(config)
    clf_final.fit(X_all, y)

    train_time = pd.Timestamp.now() - start_time
    print(f"Trained final model on ALL data ({len(df)} rows) in {train_time}")

    # Save feature importances
    try:
        save_feature_importances(clf_final, X_all, y, features, config.output_dir)
    except Exception as e:
        print(f"[WARN] feature importance failed: {e}")

    return clf_final, train_time


def _save_final_results(config: WinnerClassifierConfig, final_model, features: List[str],
                       df: pd.DataFrame, proba_oof: np.ndarray, y: np.ndarray,
                       roc_auc: float, pr_auc: float, split_kind: str):
    """Save final model pack and metrics."""
    # Find best F1 threshold
    precision, recall, thresholds = precision_recall_curve(y, proba_oof)
    best_f1, best_thr = -1.0, 0.5
    for idx, thr in enumerate(thresholds):
        if idx % 10 != 0 and idx != len(thresholds) - 1:
            continue
        yhat = (proba_oof >= thr).astype(int)
        f1 = f1_score(y, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    cm = confusion_matrix(y, (proba_oof >= best_thr).astype(int))
    tn, fp, fn, tp = cm.ravel()

    # Compile metrics
    metrics = {
        "roc_auc_oof": float(roc_auc),
        "pr_auc_oof": float(pr_auc),
        "best_f1_threshold_oof": float(best_thr),
        "best_f1_oof": float(best_f1),
        "coverage_at_best_f1_oof": float((proba_oof >= best_thr).mean()),
        "confusion_at_best_f1_oof": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "features": features,
        "impute_missing": config.impute_missing,
        "use_weights": config.use_weights,
        "cv": {"kind": split_kind, "folds": config.oof_folds}
    }

    # Save metrics
    with open(os.path.join(config.output_dir, "winner_classifier_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model pack
    pack = {
        "model": final_model,
        "model_type": config.model_type,
        "features": features,
        "medians": (DataPreprocessor(config).compute_medians(df, features) if config.impute_missing else None),
        "impute_missing": config.impute_missing,
        "targets_table_path": os.path.join(config.output_dir, "threshold_table.csv"),
        "metrics": metrics,
        "label": f"{config.train_target} > 0",
        "oof_scores_path": os.path.join(config.output_dir, "winner_scores_oof.csv"),
    }

    model_filename = f"{config.model_name}_{config.model_type}.pkl"
    joblib.dump(pack, os.path.join(config.output_dir, model_filename))

if __name__ == "__main__":
    main()
