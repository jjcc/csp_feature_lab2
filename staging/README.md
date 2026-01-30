# Staging - Old Model Experiments

This folder contains old model training and scoring scripts that are no longer actively used. The project is now focused on the **Winner Classifier** only.

## Moved: 2026-01-30

### Tail Risk Models (September 2025)
- `train_tail_with_gex.py` - Train model to predict worst K% trades by PnL
- `score_tail_with_gex.py` - Score new trades for tail risk

### Rescue Models (August-September 2025)
- `train_rescue_model.py` - Original rescue model for false negatives
- `train_rescue_model_monocal.py` - Monotonic calibration variant
- `score_rescue_model.py` - Score trades using rescue model
- `apply_rescue.py` - Apply rescue logic to existing predictions

### Regressor (August 2025)
- `train_winner_regressor.py` - Regression model to estimate return magnitude

### Meta/Accept Models (September 2025)
- `train_accept_meta_env.py` - Meta-model for candidate acceptance

### Old Winner Classifier (September 2025)
- `train_winner_classifier_pct.py` - Non-OOF version, superseded by `b01train_winner_classifier_pct_oof.py`

### Old Evaluation Scripts (August 2025)
- `evaluate_csp_filters.py` - Old filter evaluation script

### Hybrid Experiments (August 2025)
- `a11hybrid_runner_env.py` - Combined multiple models in production pipeline

## Current Active Scripts (Root Directory)

**Winner Classifier Only:**
- `b01train_winner_classifier_pct_oof.py` - Main training script with OOF CV
- `b01train_winner_classifier_pct_oof_fix.py` - Fix version with improvements
- `score_winner_classifier_env.py` - Score new candidates
- `task_score_tail_winner.py` - Production scoring pipeline
- `eval_binary_classifier_env.py` - Model evaluation metrics

## Rationale

These models were experimental approaches tried during development:
1. **Tail models**: Attempted to catch worst-case losses separately
2. **Rescue models**: Attempted to recover false negatives from winner classifier
3. **Regressor**: Attempted to estimate return magnitude instead of binary classification
4. **Meta models**: Attempted ensemble/stacking approaches

The current focus is on a single, well-tuned **Winner Classifier** that predicts profitable trades using:
- Out-of-fold cross-validation
- Multiple model types (LightGBM, CatBoost, RandomForest)
- Proper sample weighting
- Threshold calibration

## Restoring Old Models

If you need to restore any of these experiments:
1. Copy the file back to the root directory
2. Update any hard-coded paths to use pathlib (see Tier 1 fixes)
3. Ensure required columns exist in your data
4. Check that config.yaml has the necessary sections

## Cleanup

These files can be deleted if:
- No plans to revisit these model types
- Winner classifier performance is satisfactory
- Git history preserved for reference
