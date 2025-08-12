import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv; load_dotenv()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def parse_bool(x: str, default=False):
    if x is None: return default
    return str(x).strip().lower() in {"1","true","yes","y","on"}

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["win","return_pct"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")
    return df

def get_features(df: pd.DataFrame, feature_list: str):
    if feature_list is None or len(feature_list.strip()) == 0:
        default = [
            "moneyness","percentToBreakEvenBid","impliedVolatilityRank1y","delta",
            "potentialReturn","potentialReturnAnnual","breakEvenProbability",
            "openInterest","volume","underlyingLastPrice","strike"
        ]
        return [c for c in default if c in df.columns]
    else:
        feats = [c.strip() for c in feature_list.split(",") if c.strip() in df.columns]
        if not feats:
            raise SystemExit("No valid features from FEATURES env var were found in the CSV.")
        return feats

def main():
    CSV = os.getenv("CSV", "labeled_trades.csv")
    OUTPUT_DIR = ensure_dir(os.getenv("OUTPUT_DIR", "./output_winner"))
    FEATURES = os.getenv("FEATURES", "")
    USE_WEIGHTED_CLASSIFIER = parse_bool(os.getenv("USE_WEIGHTED_CLASSIFIER", "true"))
    TEST_SIZE = float(os.getenv("TEST_SIZE", "0.30"))
    CLASSIFIER_N_EST = int(os.getenv("CLASSIFIER_N_ESTIMATORS", "500"))

    df = load_dataset(CSV)
    feats = get_features(df, FEATURES)

    X = df[feats].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["win"], errors="coerce")

    if USE_WEIGHTED_CLASSIFIER:
        ret = pd.to_numeric(df["return_pct"], errors="coerce").fillna(0.0)
        w = 1.0 + 0.02 * ret.abs()
        w = w.clip(0.5, 10.0)
    else:
        w = None

    mask = (~X.isna().any(axis=1)) & (~y.isna())
    X, y = X[mask], y[mask]
    if w is not None:
        w = w[mask]

    stratify = y if y.nunique() > 1 else None
    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X, y, w, test_size=TEST_SIZE, random_state=42, stratify=stratify
    )

    clf = RandomForestClassifier(
        n_estimators=CLASSIFIER_N_EST,
        random_state=42,
        class_weight="balanced" if not USE_WEIGHTED_CLASSIFIER else None,
        n_jobs=-1
    )
    clf.fit(X_tr, y_tr, sample_weight=w_tr if USE_WEIGHTED_CLASSIFIER else None)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    # Metrics
    report = classification_report(y_te, y_pred, zero_division=0, output_dict=True)
    auc = roc_auc_score(y_te, y_prob)
    with open(os.path.join(OUTPUT_DIR, "ml_classifier_metrics.json"), "w") as f:
        json.dump({"classification_report": report, "ROC_AUC": auc, "features": feats}, f, indent=2)

    # Precision–Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_te, y_prob)
    coverage = [ (y_prob >= t).mean() for t in thresholds ]
    pr_df = pd.DataFrame({"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1], "coverage": coverage})
    pr_df.to_csv(os.path.join(OUTPUT_DIR, "precision_recall_coverage.csv"), index=False)

    plt.figure(figsize=(8,6))
    plt.plot(coverage, precision[:-1], label="Precision vs Coverage")
    plt.plot(coverage, recall[:-1], label="Recall vs Coverage")
    plt.xlabel("Coverage (fraction of trades kept)")
    plt.ylabel("Score")
    plt.title("Precision & Recall vs Coverage — Winner Classifier")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "precision_recall_coverage.png"), dpi=150)
    plt.close()

    # Save model
    import joblib
    joblib.dump(clf, os.path.join(OUTPUT_DIR, "winner_classifier.pkl"))

    print(f"✅ Winner classifier trained. AUC={auc:.4f}")
    print(f"Metrics, CSV & chart saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
