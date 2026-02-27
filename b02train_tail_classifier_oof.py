#!/usr/bin/env python3
"""
Train tail classifier to detect catastrophic bin-0 trades (left-tail losses).

This addresses the toxic trade problem where 646 out of 2505 predicted bin-3
trades are actually bin-0. The tail classifier provides a veto layer to filter
out high-risk trades before execution.

Usage:
    python b02train_tail_classifier_oof.py
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.isotonic import IsotonicRegression

from service.env_config import getenv


def main():
    """Train tail classifier using OOF splits from 4-bin model."""

    print("="*60)
    print("TAIL CLASSIFIER TRAINING (OOF)")
    print("="*60)

    # Load OOF predictions from 4-bin model
    scores_path = getenv("WINNER_OUTPUT_DIR", "output/winner_train/v9_oof_origorig") + "/winner_scores_oof.csv"
    trades_path = getenv("COMMON_OUTPUT_DIR") + "/" + getenv("COMMON_LABELED") + "/" + getenv("COMMON_OUTPUT_CSV")
    output_dir = getenv("TAIL_OUTPUT_DIR", "output/tail_train/v1_oof_orig")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"{output_dir}_{run_ts}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Loading OOF scores: {scores_path}")
    scores = pd.read_csv(scores_path)

    print(f"[INFO] Loading labeled trades: {trades_path}")
    trades = pd.read_csv(trades_path)

    # Merge on row_idx
    df = trades.merge(scores[['row_idx', 'y_true', 'y_pred', 'p_bin0', 'p_bin1', 'p_bin2', 'p_bin3', 'fold', 'has_oof']],
                     left_index=True, right_on='row_idx', how='inner')

    # Filter to valid OOF only
    df = df[df['has_oof'] == 1].copy()

    print(f"[INFO] Valid OOF trades: {len(df)}")

    # Define tail label: y_true == 0 (worst bin)
    df['tail'] = (df['y_true'] == 0).astype(int)

    print(f"\n[INFO] Tail label distribution:")
    print(f"  Tail (bin-0): {df['tail'].sum()} ({df['tail'].mean():.1%})")
    print(f"  Non-tail: {(~df['tail'].astype(bool)).sum()} ({(1-df['tail'].mean()):.1%})")

    # Analyze toxic trades (predicted bin-3, actual bin-0)
    toxic = df[(df['y_pred'] == 3) & (df['y_true'] == 0)]
    pred_bin3 = df[df['y_pred'] == 3]

    print(f"\n[INFO] Toxic trade analysis:")
    print(f"  Predicted bin-3: {len(pred_bin3)}")
    print(f"  Toxic (pred=3, true=0): {len(toxic)}")
    print(f"  Contamination rate: {len(toxic)/len(pred_bin3):.1%}")

    # Features for tail classifier
    from service.utils import BASE_FEATS, GEX_FEATS, NEW_FEATS

    # Start with all features
    all_features = BASE_FEATS + NEW_FEATS + ["gex_neg", "gex_center_abs_strike", "gex_total_abs"]
    all_features = [f for f in all_features if f in df.columns]

    # Add 4-bin model probabilities as features
    prob_features = ['p_bin0', 'p_bin1', 'p_bin2', 'p_bin3']

    # Add conflict score
    df['conflict_score'] = df['p_bin0'] * df['p_bin3']
    prob_features.append('conflict_score')

    features = all_features + prob_features

    print(f"\n[INFO] Using {len(features)} features")
    print(f"  Base features: {len(all_features)}")
    print(f"  Probability features: {len(prob_features)}")

    # Prepare data
    X = df[features].fillna(df[features].median())
    y = df['tail'].values

    # OOF predictions (reuse fold structure from 4-bin model)
    oof_tail = np.zeros(len(df))

    print(f"\n[INFO] Training tail classifier with fold-based OOF...")

    fold_metrics = []

    for fold in sorted(df['fold'].unique()):
        if fold == -1:
            continue

        print(f"  Fold {fold}...")

        train_mask = df['fold'] != fold
        val_mask = df['fold'] == fold

        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]

        # Calculate scale_pos_weight for class imbalance
        pos = y_train.sum()
        neg = len(y_train) - pos
        scale_pos_weight = neg / max(pos, 1)

        print(f"    Train: {len(y_train)} trades ({y_train.mean():.1%} tail)")
        print(f"    Val: {len(y_val)} trades ({y_val.mean():.1%} tail)")
        print(f"    scale_pos_weight: {scale_pos_weight:.2f}")

        # Train LightGBM
        model = LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            reg_alpha=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[],
        )

        # Predict
        oof_tail[val_mask] = model.predict_proba(X_val)[:, 1]

        # Fold metrics
        auc = roc_auc_score(y_val, oof_tail[val_mask])
        pr_auc = average_precision_score(y_val, oof_tail[val_mask])

        fold_metrics.append({
            'fold': int(fold),
            'auc': float(auc),
            'pr_auc': float(pr_auc),
        })

        print(f"    ROC-AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")

    df['p_tail_oof'] = oof_tail

    # Overall OOF evaluation
    print(f"\n[INFO] Overall OOF Evaluation:")

    valid = df['p_tail_oof'] > 0
    auc = roc_auc_score(y[valid], oof_tail[valid])
    pr_auc = average_precision_score(y[valid], oof_tail[valid])

    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")

    # Calibrate probabilities using Isotonic Regression
    print(f"\n[INFO] Calibrating tail probabilities...")

    cal = IsotonicRegression(out_of_bounds='clip')
    cal.fit(df.loc[valid, 'p_tail_oof'], df.loc[valid, 'tail'])
    df['p_tail_cal'] = cal.transform(df['p_tail_oof'])

    # Evaluate on toxic slice (predicted bin-3, actual bin-0)
    print(f"\n[INFO] Evaluation on Toxic Slice (pred=3, true=0):")

    mask_toxic = (df['y_pred'] == 3) & (df['y_true'] == 0)
    mask_pred3 = (df['y_pred'] == 3)

    for threshold in [0.25, 0.30, 0.35, 0.40, 0.45]:
        # How many toxic trades would we catch?
        toxic_caught = df[mask_toxic & (df['p_tail_cal'] >= threshold)]
        recall_toxic = len(toxic_caught) / mask_toxic.sum()

        # How many predicted bin-3 trades would we flag?
        flagged = df[mask_pred3 & (df['p_tail_cal'] >= threshold)]
        flag_rate = len(flagged) / mask_pred3.sum()

        # Of flagged trades, how many are truly tail?
        precision = flagged['tail'].mean() if len(flagged) > 0 else 0

        print(f"  Threshold {threshold:.2f}:")
        print(f"    Recall on toxic: {recall_toxic:.1%} ({len(toxic_caught)}/{mask_toxic.sum()})")
        print(f"    Flag rate on pred-3: {flag_rate:.1%} ({len(flagged)}/{mask_pred3.sum()})")
        print(f"    Precision: {precision:.1%}")

    # Train final model on all data
    print(f"\n[INFO] Training final model on all data...")

    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = neg / max(pos, 1)

    final_model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        reg_alpha=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    final_model.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n[INFO] Top 15 features:")
    print(importance.head(15).to_string(index=False))

    # Save model and results
    import joblib

    model_path = Path(output_dir) / "tail_classifier_model.pkl"
    joblib.dump(final_model, model_path)
    print(f"\n[INFO] Saved model: {model_path}")

    # Save calibrator
    cal_path = Path(output_dir) / "tail_calibrator.pkl"
    joblib.dump(cal, cal_path)
    print(f"[INFO] Saved calibrator: {cal_path}")

    # Save OOF predictions
    oof_df = pd.DataFrame({
        'row_idx': df['row_idx'],
        'fold': df['fold'],
        'y_true': df['tail'],
        'y_pred_bin': df['y_pred'],
        'p_tail_oof': df['p_tail_oof'],
        'p_tail_cal': df['p_tail_cal'],
        'is_toxic': mask_toxic,
        'return_mon': df['return_mon'],
    })
    oof_path = Path(output_dir) / "tail_oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"[INFO] Saved OOF: {oof_path}")

    # Save metrics
    metrics = {
        'approach': 'tail_binary_bin0',
        'roc_auc': float(auc),
        'pr_auc': float(pr_auc),
        'n_features': len(features),
        'features': features,
        'fold_metrics': fold_metrics,
        'toxic_stats': {
            'total_pred_bin3': int(mask_pred3.sum()),
            'toxic_trades': int(mask_toxic.sum()),
            'contamination_rate': float(mask_toxic.sum() / mask_pred3.sum()),
        },
    }

    metrics_path = Path(output_dir) / "tail_classifier_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved metrics: {metrics_path}")

    # Save importance
    importance_path = Path(output_dir) / "tail_feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    print(f"[INFO] Saved importance: {importance_path}")

    print(f"\n{'='*60}")
    print("DONE")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. Review feature importance: {importance_path}")
    print(f"  2. Analyze OOF predictions: {oof_path}")
    print(f"  3. Run combined policy evaluation: python analyze_tail_combined_policy.py")


if __name__ == "__main__":
    main()
