# Alternative Approaches to 4-Bin Classification

## Context

You discovered:
1. **Binary classification fails**: ROC-AUC 0.545 (barely better than random)
2. **4-bin classification works**: 47% accuracy, 71% top-bin recall
3. **But**: Results didn't improve after fixing return_mon alignment
4. **Root cause**: Most trades are 3-4 DTE, so return_pct ≈ return_mon

**Key insight**: Features predict **relative ranking**, not absolute outcomes.

## Alternative Approaches

### 1. Direct Regression (Predict Continuous Returns)

**Instead of**: Predicting discrete bins
**Try**: Predicting actual return values

```python
# In training script
y = df['return_pct']  # Or return_mon - doesn't matter

# Use regression model
from lightgbm import LGBMRegressor
model = LGBMRegressor(
    objective='regression',
    metric='rmse',  # Or 'mae', 'huber'
    n_estimators=2000,
    learning_rate=0.05,
    # ... other params
)

model.fit(X, y, sample_weight=weights)

# At inference
predicted_return = model.predict(X_new)

# Trading strategy: Take top N by predicted return
trades = df.nlargest(50, 'predicted_return')
```

**Advantages:**
- Captures full information (not discretized into bins)
- Can optimize for actual return, not just ranking
- More flexible for portfolio optimization

**Disadvantages:**
- Harder to evaluate (what's "good" RMSE for returns?)
- May overfit to outliers
- Predictions might not be well-calibrated

**Try this:**
```bash
# Modify b01 to use LGBMRegressor instead of LGBMClassifier
# Or create new script: b01train_winner_regressor.py
```

### 2. Learning to Rank (Pairwise/Listwise)

**Instead of**: Predicting labels or values
**Try**: Learning which trade is better than another

```python
# LightGBM has native ranking support
from lightgbm import LGBMRanker

# Group by trade_date (rank within each day)
groups = df.groupby('trade_date').size().values

model = LGBMRanker(
    objective='lambdarank',  # Or 'rank_xendcg'
    metric='ndcg',
    n_estimators=2000,
)

model.fit(X, y, group=groups)

# At inference
scores = model.predict(X_new)
# Higher score = better trade
trades = df.nlargest(50, 'score')
```

**Advantages:**
- Directly optimizes for ranking quality (NDCG)
- Natural fit for "pick best N trades" problem
- Handles ties and relative preferences well

**Disadvantages:**
- Requires grouping (trade_date in your case)
- Less interpretable than classification
- Fewer evaluation tools

**This might be your best bet!** It's explicitly designed for the problem you're solving.

### 3. Ordinal Regression

**Instead of**: Treating bins as nominal (unordered)
**Try**: Treating bins as ordinal (ordered: 0 < 1 < 2 < 3)

```python
# Option A: Use ordinal-specific loss
# LightGBM doesn't have native ordinal support, but you can approximate

# Option B: Cumulative link models (requires mord package)
from mord import LogisticAT

model = LogisticAT(alpha=1.0)
model.fit(X, y)

# Option C: Use regression but round to bins
from lightgbm import LGBMRegressor

model = LGBMRegressor(...)
model.fit(X, y_continuous)  # Train on 0-3 as continuous

# At inference
predictions_continuous = model.predict(X_new)
predictions_bins = np.clip(np.round(predictions_continuous), 0, 3).astype(int)
```

**Advantages:**
- Respects ordering (confusing bin 0 with bin 3 is worse than with bin 1)
- Can reduce extreme misclassifications
- More sample-efficient than nominal classification

**Disadvantages:**
- Fewer tools available
- Harder to implement and tune

### 4. Two-Stage Classifier

**Instead of**: One 4-way classifier
**Try**: Two binary classifiers in sequence

```python
# Stage 1: Separate extremes from middle
# Bin 0+1 (below median) vs Bin 2+3 (above median)
y_stage1 = (y >= 2).astype(int)
model_stage1 = LGBMClassifier(...)
model_stage1.fit(X, y_stage1)

# Stage 2a: Within below-median, separate bin 0 vs 1
mask_below = (y < 2)
model_stage2_below = LGBMClassifier(...)
model_stage2_below.fit(X[mask_below], y[mask_below])

# Stage 2b: Within above-median, separate bin 2 vs 3
mask_above = (y >= 2)
model_stage2_above = LGBMClassifier(...)
model_stage2_above.fit(X[mask_above], y[mask_above] - 2)  # Map to 0/1

# At inference
pred_stage1 = model_stage1.predict(X_new)
if pred_stage1 == 0:  # Below median
    pred_stage2 = model_stage2_below.predict(X_new)
else:  # Above median
    pred_stage2 = model_stage2_above.predict(X_new) + 2
```

**Advantages:**
- Can use different features for different stages
- Reduces bin 0 ↔ bin 3 confusion
- More interpretable

**Disadvantages:**
- More complex to train and deploy
- Errors compound across stages

### 5. Ensemble of Approaches

**Instead of**: Picking one method
**Try**: Combining multiple methods

```python
# Train multiple models
model_4bin = train_4bin_classifier(X, y)
model_reg = train_regressor(X, y_continuous)
model_rank = train_ranker(X, y, groups)

# At inference, combine predictions
score_4bin = model_4bin.predict_proba(X)[:, 3]  # Prob of top bin
score_reg = model_reg.predict(X)  # Predicted return
score_rank = model_rank.predict(X)  # Ranking score

# Weighted average
final_score = 0.4 * normalize(score_4bin) + \
              0.3 * normalize(score_reg) + \
              0.3 * normalize(score_rank)

# Or: Vote (pick trades that all models agree on)
top_by_4bin = set(df.nlargest(100, 'score_4bin').index)
top_by_reg = set(df.nlargest(100, 'score_reg').index)
top_by_rank = set(df.nlargest(100, 'score_rank').index)

# Intersection: Trades all models like
high_conviction = top_by_4bin & top_by_reg & top_by_rank
```

**Advantages:**
- Robust to individual model failures
- Can capture different aspects of the problem
- Often outperforms single models

**Disadvantages:**
- More complex to maintain
- Requires careful tuning of weights

## Feature Engineering for Ranking

Since features predict ranking better than outcomes, add more **relative** features:

```python
# Relative to daily distribution
df['delta_rank'] = df.groupby('trade_date')['delta'].rank(pct=True)
df['ivr_rank'] = df.groupby('trade_date')['impliedVolatilityRank1y'].rank(pct=True)
df['premium_rank'] = df.groupby('trade_date')['entry_credit'].rank(pct=True)

# Relative to daily mean
df['delta_vs_mean'] = df.groupby('trade_date')['delta'].transform(
    lambda x: x - x.mean()
)

# Extremeness indicators
df['is_otm_extreme'] = (df['moneyness'] < 0.95) | (df['moneyness'] > 1.05)
df['is_high_vol_day'] = df.groupby('trade_date')['VIX'].transform(
    lambda x: x.iloc[0] > x.mean()  # Today's VIX vs historical mean
)

# Interaction: Does delta matter more in high-vol?
df['delta_x_vix'] = df['delta'] * df['VIX']

# Comparative features
df['premium_to_strike_ratio'] = df['entry_credit'] / df['strike']
df['volume_to_oi_ratio'] = df['volume'] / (df['openInterest'] + 1)
```

## Evaluation Metrics for Ranking

**Instead of**: Accuracy, precision, recall
**Try**: Ranking-specific metrics

```python
from sklearn.metrics import ndcg_score, dcg_score

# Normalized Discounted Cumulative Gain (NDCG)
# Measures quality of ranking
# 1.0 = perfect ranking, 0.0 = random

y_true = df['return_pct'].values  # Actual returns
y_pred = df['predicted_score'].values  # Model scores

ndcg = ndcg_score([y_true], [y_pred])
print(f"NDCG: {ndcg:.3f}")

# Or: Spearman correlation (rank correlation)
from scipy.stats import spearmanr

corr, pval = spearmanr(y_true, y_pred)
print(f"Spearman: {corr:.3f}")

# Or: Top-K precision
# Of top K predicted trades, what's their mean return?
k = 50
top_k = df.nlargest(k, 'predicted_score')
top_k_return = top_k['return_pct'].mean()
print(f"Top-{k} mean return: {top_k_return:.2f}%")
```

## Recommended Next Steps

### Priority 1: Try Regression

**Easiest to implement**, gives you continuous predictions:

```bash
# Create b01train_winner_regressor.py
# Copy b01train_winner_classifier_bins4_oof.py
# Replace LGBMClassifier with LGBMRegressor
# Change objective='regression', metric='rmse'
# Remove bin-related code

python b01train_winner_regressor.py

# Compare top-N returns
# Regression vs 4-bin vs binary
```

**Expected**: Similar or better performance, more flexible

### Priority 2: Try LGBMRanker

**Most appropriate for your problem**:

```python
# In new script: b01train_winner_ranker.py
from lightgbm import LGBMRanker

# Group by trade_date for within-day ranking
df['trade_date'] = pd.to_datetime(df['tradeTime']).dt.date
df = df.sort_values('trade_date')
groups = df.groupby('trade_date').size().values

model = LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
)

# Labels should be actual returns (higher = better)
y = df['return_pct'].values

model.fit(X, y, group=groups)

# At inference, rank by predicted score
scores = model.predict(X_test)
best_trades = test_df.nlargest(50, 'score')
```

### Priority 3: Add Relative Features

**Improve any model** with ranking-focused features:

```python
# Add these to your feature engineering
df['delta_rank_daily'] = df.groupby('trade_date')['delta'].rank(pct=True)
df['premium_rank_daily'] = df.groupby('trade_date')['entry_credit'].rank(pct=True)
df['ivr_rank_daily'] = df.groupby('trade_date')['impliedVolatilityRank1y'].rank(pct=True)
```

## Expected Results

**Current (4-bin)**:
- Accuracy: 47%
- Top-bin recall: 71%
- Top-10% mean return: ~0-2% (marginal)

**After Regression**:
- Top-10% mean return: **3-5%** (better selection)
- RMSE: ~8-12% (hard to interpret)
- More flexible for position sizing

**After LGBMRanker**:
- NDCG: 0.55-0.65 (vs 0.5 random)
- Top-10% mean return: **4-6%** (best selection)
- Directly optimized for your use case

**After Relative Features**:
- Any model: **+1-2% accuracy**
- Better separation of extremes
- More stable across time periods

---

## Quick Start: Try Regression

```bash
# 1. Create regression version
cp b01train_winner_classifier_bins4_oof.py b01train_winner_regressor.py

# 2. Edit b01train_winner_regressor.py
# Replace:
#   from lightgbm import LGBMClassifier
#   model = LGBMClassifier(objective='multiclass', ...)
# With:
#   from lightgbm import LGBMRegressor
#   model = LGBMRegressor(objective='regression', metric='rmse', ...)

# 3. Train
python b01train_winner_regressor.py

# 4. Compare
python eval_regression_vs_classification.py
```

Would you like me to create a regression version of the training script for you?
