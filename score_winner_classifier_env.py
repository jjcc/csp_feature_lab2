#!/usr/bin/env python3
"""
score_winner_classifier_env.py

Score the input data using the winner classifier model.

Note:
Why I need score_winner_classifier_env.py?
This script  scores data using the winner classifier model separately. With another evaluation script I can compare the seperate scoring functions
behave the same way as the training so the training performance report is applicable to another separate scoring function in this script.
 *(refactored to use model_utils)
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

from sklearn.metrics import average_precision_score, roc_auc_score
from service.utils import load_env_default, ensure_dir, prep_winner_like_training, pick_threshold_auto
from service.preprocess import add_dte_and_normalized_returns


@dataclass
class ScoringConfig:
    """Configuration for winner classifier scoring."""
    csv_in: str
    model_in: str
    csv_out_dir: str
    csv_out: str
    proba_col: str
    pred_col: str
    train_target: str
    gex_filter: bool
    use_other_model: bool
    model_type: str
    fixed_threshold: Optional[float]
    use_pack_f1: bool
    target_precisions: List[float]
    target_recalls: List[float]
    auto_calibrate: bool
    split_file: str
    use_oof: bool
    train_epsilon: float
    write_sweep: bool


def load_scoring_config() -> ScoringConfig:
    """Load and validate configuration from environment variables."""
    load_env_default()

    csv_in = os.getenv("WINNER_SCORE_INPUT", "./candidates.csv")
    gex_filter = str(os.getenv("FILTER_GEX", "0")).lower() in {"1", "true", "yes", "y", "on"}

    use_other_model = False  # TODO: Make this configurable

    if use_other_model:
        model_in = os.getenv("WINNER_MODEL_IN", "./output_winner/model_pack.pkl")
        model_type = ""
    else:
        model_type = os.getenv("WINNER_MODEL_TYPE", "lgbm").strip().lower()
        model_in = os.path.join(
            os.getenv("WINNER_OUTPUT_DIR", "output"),
            f"{os.getenv('WINNER_MODEL_NAME')}_{model_type}.pkl"
        )

    csv_out_dir = os.getenv("WINNER_SCORE_OUT_FOLDER", "output/winner_score/folder1")
    csv_out = os.path.join(csv_out_dir, os.getenv("WINNER_SCORE_OUT", "scores_winner.csv"))

    fixed_thr = os.getenv("WINNER_SCORE_THRESHOLD", "").strip()
    fixed_threshold = float(fixed_thr) if fixed_thr else None

    target_prec = os.getenv("WINNER_SCORE_TARGET_PRECISION", "").strip()
    target_recall = os.getenv("WINNER_SCORE_TARGET_RECALL", "").strip()

    target_precisions = [float(x.strip()) for x in target_prec.split(",") if x.strip()] if target_prec else []
    target_recalls = [float(x.strip()) for x in target_recall.split(",") if x.strip()] if target_recall else []

    split_file = os.getenv("WINNER_SPLIT_FILE", "").strip()
    if split_file:
        split_file = os.path.join(os.getenv("WINNER_OUTPUT_DIR", "output"), split_file)

    return ScoringConfig(
        csv_in=csv_in,
        model_in=model_in,
        csv_out_dir=csv_out_dir,
        csv_out=csv_out,
        proba_col=os.getenv("WINNER_PROBA_COL", "prob_winner"),
        pred_col=os.getenv("WINNER_PRED_COL", "pred_winner"),
        train_target=os.getenv("WINNER_TRAIN_TARGET", "return_mon").strip(),
        gex_filter=gex_filter,
        use_other_model=use_other_model,
        model_type=model_type,
        fixed_threshold=fixed_threshold,
        use_pack_f1=str(os.getenv("WINNER_SCORE_USE_PACK_BEST_F1", "1")).lower() in {"1", "true", "yes", "y", "on"},
        target_precisions=target_precisions,
        target_recalls=target_recalls,
        auto_calibrate=str(os.getenv("WINNER_SCORE_AUTO_CALIBRATE", "0")).lower() in {"1", "true", "yes", "y", "on"},
        split_file=split_file,
        use_oof=True,  # TODO: Make this configurable
        train_epsilon=float(os.getenv("WINNER_TRAIN_EPSILON", "0.00")),
        write_sweep=str(os.getenv("WRITE_SWEEP", "1")).lower() in {"1", "true", "yes", "y", "on"}
    )




def pick_threshold_from_coverage(proba: np.ndarray, coverage: float) -> float:
    """Pick threshold to achieve target coverage."""
    if len(proba) == 0:
        return 1.0
    k = max(1, int(round(len(proba) * coverage)))
    thr = np.partition(proba, len(proba) - k)[len(proba) - k]
    return float(thr)


def load_and_preprocess_data(config: ScoringConfig) -> pd.DataFrame:
    """Load and preprocess input data."""
    df = pd.read_csv(config.csv_in)

    if config.gex_filter and "gex_missing" in df.columns:
        df = df[df["gex_missing"] == 0].copy()
        print(f"Filtered rows with missing GEX, remaining {len(df)} rows.")

    df = add_dte_and_normalized_returns(df)

    if "tradeTime" in df.columns:
        df["tradeTime"] = pd.to_datetime(df["tradeTime"], errors="coerce")

    # Apply train/test split filtering
    if config.split_file and not config.use_oof:
        if not os.path.isfile(config.split_file):
            raise FileNotFoundError(f"WINNER_SPLIT_FILE not found: {config.split_file}")

        df_split = pd.read_csv(config.split_file)
        if "tradeTime" in df_split.columns:
            df_split["tradeTime"] = pd.to_datetime(df_split["tradeTime"], errors="coerce")

        df = df.merge(df_split, on=["symbol", "tradeTime"], how="left")

        # Clean up duplicate columns from merge
        col_x = [col for col in df.columns if col.endswith("_x")]
        for col in col_x:
            real_col = col[:-2]
            df[real_col] = df[col]
            df = df.drop(columns=[col, real_col + "_y"])

        df = df[df["is_train"] == 0]
    else:
        if "is_train" in df.columns:
            df = df[df["is_train"] == 0]

    return df


def select_threshold(config: ScoringConfig, proba: np.ndarray, y: Optional[np.ndarray], best_f1_thr: float) -> Tuple[float, Optional[pd.DataFrame]]:
    """Select threshold based on configuration."""
    if config.fixed_threshold is not None:
        return config.fixed_threshold, None

    if config.auto_calibrate and y is not None and (config.target_precisions or config.target_recalls):
        return pick_threshold_auto(y, proba, config.target_precisions, config.target_recalls)

    if config.use_pack_f1:
        return best_f1_thr, None

    return 0.5, None


def calculate_metrics(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
    """Calculate AUC-ROC and AUC-PRC metrics."""
    auc_roc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float('nan')
    auc_prc = average_precision_score(y_true, proba)
    return auc_roc, auc_prc


def write_threshold_sweep(proba: np.ndarray, y: Optional[np.ndarray], output_path: str) -> None:
    """Write threshold sweep analysis."""
    coverages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    rows = []

    for cov in coverages:
        thr = pick_threshold_from_coverage(proba, cov)
        mask = proba >= thr
        row = {"coverage": cov, "threshold": thr, "n": int(mask.sum())}

        if y is not None:
            row["precision_est"] = float(y[mask].mean()) if mask.any() else np.nan
            row["recall_est"] = float((y[mask] == 1).sum() / max(1, (y == 1).sum()))

        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path.replace("_scored.csv", "_threshold_sweep.csv"), index=False)


def write_outputs(config: ScoringConfig, out: pd.DataFrame, chosen_thr: float, y: Optional[np.ndarray],
                 proba: np.ndarray, thr_table: Optional[pd.DataFrame]) -> None:
    """Write all output files."""
    os.makedirs(config.csv_out_dir, exist_ok=True)
    ensure_dir(config.csv_out)
    out.to_csv(config.csv_out, index=False)

    # Write metrics if we have labels
    if y is not None:
        auc_roc, auc_prc = calculate_metrics(y, proba)
        print(f"AUC-ROC: {auc_roc:.4f}, AUC-PRC: {auc_prc:.4f}")

        with open(Path(config.csv_out).with_suffix(".metrics.txt"), "w") as f:
            f.write(f"AUC-ROC: {auc_roc:.6f}\nAUC-PRC: {auc_prc:.6f}\n")

    # Write summary
    summary = {
        "rows_scored": int(len(out)),
        "threshold": float(chosen_thr),
        "predicted_winners": int(out[config.pred_col].sum()),
        "coverage": float(out[config.pred_col].mean()),
    }
    with open(Path(config.csv_out).with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Write threshold table if available
    if thr_table is not None:
        thr_csv = Path(config.csv_out).with_name(Path(config.csv_out).stem + "_threshold_table.csv")
        thr_table.to_csv(thr_csv, index=False)

    # Write threshold sweep analysis
    if config.write_sweep:
        out_scored = os.getenv("OUT_SCORED", config.csv_out.replace(".csv", "_scored.csv"))
        write_threshold_sweep(proba, y, out_scored)


def main():
    """Main function to score winner classifier."""
    config = load_scoring_config()

    # Load model
    pack = joblib.load(config.model_in)
    print(f"Loaded model from {config.model_in}")
    clf = pack["model"]
    feats = pack["features"]
    medians = pack.get("medians", None)
    impute_missing = bool(pack.get("impute_missing", bool(medians is not None)))
    best_f1_thr = float(pack.get("metrics", {}).get("best_f1_threshold", 0.5))

    # Load and preprocess data
    df = load_and_preprocess_data(config)

    # Prepare features
    X, mask = prep_winner_like_training(df, feats, medians=medians, impute_missing=impute_missing)

    # Score
    proba = clf.predict_proba(X)[:, 1]
    out = df.loc[mask].copy()
    out[config.proba_col] = proba

    # Extract labels if available
    y = None
    if config.train_target in out.columns:
        y = (pd.to_numeric(out[config.train_target], errors="coerce") > config.train_epsilon).astype(int).values
    elif "win" in out.columns:
        y = out["win"].astype(int).values

    # Select threshold
    chosen_thr, thr_table = select_threshold(config, proba, y, best_f1_thr)

    # Apply threshold
    out[config.pred_col] = (out[config.proba_col] >= chosen_thr).astype(int)
    out["win_labeled"] = y if y is not None else np.nan

    # Write outputs
    write_outputs(config, out, chosen_thr, y, proba, thr_table)

    print(f"[OK] Scored {len(out)} rows. Saved → {config.csv_out}")
    print(f"Threshold={chosen_thr:.6f}, coverage={out[config.pred_col].mean():.4f} for target precision {config.target_precisions} or recall {config.target_recalls}")

if __name__ == "__main__":
    main()
