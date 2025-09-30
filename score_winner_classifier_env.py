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
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

from service.utils import load_env_default, ensure_dir
from service.preprocess import add_dte_and_normalized_returns
from service.winner_scoring import (
    load_winner_model, score_winner_data, apply_winner_threshold,
    select_winner_threshold, calculate_winner_metrics, extract_labels,
    write_winner_summary, write_winner_metrics
)
from service.production_data import add_features, parse_target_time
from service.env_config import getenv


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
    process_on_fly: bool


def load_scoring_config() -> ScoringConfig:
    """Load and validate configuration from environment variables."""

    csv_in = getenv("WINNERSCORE_SCORE_INPUT", "./candidates.csv")
    gex_filter = str(getenv("GEX_FILTER", "0")).lower() in {"1", "true", "yes", "y", "on"}

    use_other_model = False  # TODO: Make this configurable

    if use_other_model:
        model_in = getenv("WINNERSCORE_MODEL_IN", "./output_winner/model_pack.pkl")
        model_type = ""
    else:
        model_type = getenv("WINNER_MODEL_TYPE", "lgbm").strip().lower()
        model_in = os.path.join(
            getenv("WINNER_OUTPUT_DIR", "output"),
            f"{getenv('WINNER_MODEL_NAME')}_{model_type}.pkl"
        )

    csv_out_dir = getenv("WINNERSCORE_SCORE_OUT_FOLDER", "output/winner_score/folder1")
    csv_out = os.path.join(csv_out_dir, getenv("WINNERSCORE_SCORE_OUT", "scores_winner.csv"))

    fixed_thr = getenv("WINNERSCORE_THRESHOLD", "").strip()
    fixed_threshold = float(fixed_thr) if fixed_thr else None

    target_prec = getenv("WINNERSCORE_TARGET_PRECISION", "").strip()
    target_recall = getenv("WINNERSCORE_TARGET_RECALL", "").strip()

    target_precisions = [float(x.strip()) for x in target_prec.split(",") if x.strip()] if target_prec else []
    target_recalls = [float(x.strip()) for x in target_recall.split(",") if x.strip()] if target_recall else []

    split_file = getenv("WINNERSCORE_SPLIT_FILE", "").strip()
    if split_file:
        split_file = os.path.join(getenv("WINNER_OUTPUT_DIR", "output"), split_file)

    return ScoringConfig(
        csv_in=csv_in,
        model_in=model_in,
        csv_out_dir=csv_out_dir,
        csv_out=csv_out,
        proba_col=getenv("WINNERSCORE_PROBA_COL", "prob_winner"),
        pred_col=getenv("WINNERSCORE_PRED_COL", "pred_winner"),
        train_target=getenv("WINNER_TRAIN_TARGET", "return_mon").strip(),
        gex_filter=gex_filter,
        use_other_model=use_other_model,
        model_type=model_type,
        fixed_threshold=fixed_threshold,
        use_pack_f1=str(getenv("WINNER_SCORE_USE_PACK_BEST_F1", "1")).lower() in {"1", "true", "yes", "y", "on"},
        target_precisions=target_precisions,
        target_recalls=target_recalls,
        auto_calibrate=str(getenv("WINNER_SCORE_AUTO_CALIBRATE", "0")).lower() in {"1", "true", "yes", "y", "on"},
        split_file=split_file,
        use_oof=True,  # TODO: Make this configurable
        train_epsilon=float(getenv("WINNER_TRAIN_EPSILON", "0.00")),
        write_sweep=str(getenv("WRITE_SWEEP", "1")).lower() in {"1", "true", "yes", "y", "on"},
        process_on_fly=str(getenv("PROCESS_ON_FLY", "0")).lower() in {"1", "true", "yes", "y", "on"}
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
    non_labelled = False # assume we have labels
    if "label" not in config.csv_in:
        non_labelled = True

    if config.gex_filter and "gex_missing" in df.columns:
        df = df[df["gex_missing"] == 0].copy()
        print(f"Filtered rows with missing GEX, remaining {len(df)} rows.")

    df = add_dte_and_normalized_returns(df, non_labelled)

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
        metrics = calculate_winner_metrics(y, proba)
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}, AUC-PRC: {metrics['auc_prc']:.4f}")
        write_winner_metrics(config.csv_out, metrics)

    # Write summary using shared function
    metrics_for_summary = calculate_winner_metrics(y, proba) if y is not None else None
    write_winner_summary(config.csv_out, out, chosen_thr, config.pred_col, metrics_for_summary)

    # Write threshold table if available
    if thr_table is not None:
        thr_csv = Path(config.csv_out).with_name(Path(config.csv_out).stem + "_threshold_table.csv")
        thr_table.to_csv(thr_csv, index=False)

    # Write threshold sweep analysis
    if config.write_sweep:
        out_scored = config.csv_out.replace(".csv", "_scored.csv")
        write_threshold_sweep(proba, y, out_scored)


def load_data_on_fly(config: ScoringConfig) -> pd.DataFrame:
    """Load and process data on-the-fly (production style)."""
    option_file = config.csv_in

    # Extract date and time from filename (assuming format like task_score_tail_winner.py)
    # This is a simplified extraction - adjust based on your filename patterns
    filename = os.path.basename(option_file)
    if "_" in filename:
        parts = filename.replace(".csv", "").split("_")
        if len(parts) >= 3:
            target_date = parts[-3] if parts[-3].count("-") == 2 else "2025-01-01"  # fallback
            time_part = f"{parts[-2]}:{parts[-1]}" if len(parts) >= 4 else "11:00"
        else:
            target_date = "2025-01-01"  # fallback
            time_part = "11:00"
    else:
        target_date = "2025-01-01"  # fallback
        time_part = "11:00"

    target_t = parse_target_time(time_part)
    target_minutes = target_t.hour * 60 + target_t.minute

    # Use shared function from production_data.py
    df = add_features(target_minutes, option_file, target_date)

    # Apply same preprocessing as load_and_preprocess_data
    if config.gex_filter and "gex_missing" in df.columns:
        df = df[df["gex_missing"] == 0].copy()
        print(f"Filtered rows with missing GEX, remaining {len(df)} rows.")

    df = add_dte_and_normalized_returns(df)

    if "tradeTime" in df.columns:
        df["tradeTime"] = pd.to_datetime(df["tradeTime"], errors="coerce")

    # Note: On-fly processing doesn't use split files or train/test filtering
    # as it's meant for production data

    return df


def main():
    """Main function to score winner classifier."""
    config = load_scoring_config()

    # Load model using shared function
    model_pack = load_winner_model(config.model_in)
    print(f"Loaded model from {config.model_in}")

    # Load and preprocess data - use on-fly processing if configured
    if config.process_on_fly:
        print("Using on-fly data processing (production mode)")
        df = load_data_on_fly(config)
    else:
        print("Using pre-processed data (validation mode)")
        df = load_and_preprocess_data(config)

    # Score using shared function
    out, proba, _ = score_winner_data(df, model_pack, config.proba_col)

    # Extract labels if available
    y = extract_labels(out, config.train_target, config.train_epsilon)

    # Select threshold using shared function
    chosen_thr, thr_table = select_winner_threshold(
        proba, y,
        fixed_threshold=config.fixed_threshold,
        use_pack_f1=config.use_pack_f1,
        best_f1_threshold=model_pack.best_f1_threshold,
        auto_calibrate=config.auto_calibrate,
        target_precisions=config.target_precisions,
        target_recalls=config.target_recalls
    )

    # Apply threshold using shared function
    out = apply_winner_threshold(out, config.proba_col, config.pred_col, chosen_thr)
    out["win_labeled"] = y if y is not None else np.nan

    # Write outputs
    write_outputs(config, out, chosen_thr, y, proba, thr_table)

    print(f"[OK] Scored {len(out)} rows. Saved → {config.csv_out}")
    print(f"Threshold={chosen_thr:.6f}, coverage={out[config.pred_col].mean():.4f} for target precision {config.target_precisions} or recall {config.target_recalls}")

if __name__ == "__main__":
    main()
