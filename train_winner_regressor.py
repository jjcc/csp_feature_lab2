#!/usr/bin/env python3
"""
Winner Regressor Trainer
------------------------
Trains a regression model to predict `return_pct` from your labeled CSP dataset.
Outputs:
  - winner_regressor.pkl
  - ml_regressor_metrics.json
  - feature_importances_regressor.csv
  - scored_trades_regressor.csv
"""

import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv; load_dotenv()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from service.utils import ALL_FEATS

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "return_pct" not in df.columns:
        raise ValueError("CSV must contain 'return_pct' column.")
    return df

def get_features(df: pd.DataFrame, feature_list: str):
    if not feature_list or feature_list.strip() == "":
        default = ALL_FEATS
        feats = [c for c in default if c in df.columns]
    else:
        feats = [c.strip() for c in feature_list.split(",") if c.strip() in df.columns]
        if not feats:
            raise ValueError("No valid features from FEATURES env var found in CSV.")
    return feats

def train_return_regressor(df: pd.DataFrame, feats, test_size, n_estimators):
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["return_pct"], errors="coerce")

    mask = (~X.isna().any(axis=1)) & (~y.isna())
    X, y = X[mask], y[mask]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

    reg = RandomForestRegressor(
        n_estimators=int(n_estimators), random_state=42, n_jobs=-1
    )
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)

    metrics = {
        "R2": float(r2_score(y_te, y_pred)),
        "MAE": float(mean_absolute_error(y_te, y_pred)),
        "RMSE": float(mean_squared_error(y_te, y_pred)),
    }
    importances = pd.Series(reg.feature_importances_, index=X.columns).sort_values(ascending=False)

    return reg, metrics, importances

def main():
    env_file = os.environ.get("ENV_FILE", ".env_regressor")
    if os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        load_dotenv()
    
    CSV = os.getenv("REGRESSOR_CSV", "/mnt/data/labeled_trades_with_gex.csv")
    OUTPUT_DIR = ensure_dir(os.getenv("REGRESSOR_OUTPUT_DIR", "/mnt/data/output_regressor"))
    FEATURES = os.getenv("FEATURES", "")
    TEST_SIZE = float(os.getenv("TEST_SIZE", "0.30"))
    REGRESSOR_N_EST = int(os.getenv("REGRESSOR_N_ESTIMATORS", "600"))
    REGRESSOR_SPLIT_FILE = os.getenv("REGRESSOR_SPLIT_FILE")

    df = load_dataset(CSV)
    if REGRESSOR_SPLIT_FILE and os.path.exists(REGRESSOR_SPLIT_FILE):
        df_split = pd.read_csv(REGRESSOR_SPLIT_FILE)
        df = df.merge(df_split, on=["symbol", "tradeTime"])
        if "return_pct_x" in df.columns:
            df["return_pct"] = df["return_pct_x"]
            df.drop(columns=["return_pct_x", "return_pct_y"], inplace=True)
            # filter
            df = df[df["label"] == 1]



    feats = get_features(df, FEATURES)

    reg, reg_metrics, reg_importances = \
        train_return_regressor(df, feats, TEST_SIZE, REGRESSOR_N_EST)

    # Save metrics
    with open(os.path.join(OUTPUT_DIR, "ml_regressor_metrics_winner.json"), "w") as f:
        json.dump({"regression_metrics": reg_metrics, "features": feats}, f, indent=2)

    # Save importances
    reg_importances.to_csv(
        os.path.join(OUTPUT_DIR, "feature_importances_regressor_winner.csv"),
        header=["importance"]
    )

    # Save model
    try:
        import joblib
        joblib.dump(reg, os.path.join(OUTPUT_DIR, "winner_regressor.pkl"))
    except Exception as e:
        print("[WARN] Could not save model:", e)

    # Score all rows
    X_all = df[feats].apply(pd.to_numeric, errors="coerce")
    valid_mask = ~X_all.isna().any(axis=1)
    preds_df = df.copy()
    preds_df.loc[valid_mask, "ml_pred_return"] = reg.predict(X_all[valid_mask])
    preds_df.to_csv(os.path.join(OUTPUT_DIR, "scored_trades_regressor.csv"), index=False)

    print("Saved:")
    print(" -", os.path.join(OUTPUT_DIR, "ml_regressor_metrics.json"))
    print(" -", os.path.join(OUTPUT_DIR, "feature_importances_regressor.csv"))
    print(" -", os.path.join(OUTPUT_DIR, "winner_regressor.pkl"))
    print(" -", os.path.join(OUTPUT_DIR, "scored_trades_regressor.csv"))

if __name__ == "__main__":
    main()
