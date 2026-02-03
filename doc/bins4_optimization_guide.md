# 4-Bin Classification Optimization Guide

Since 4-bin classification shows promise where binary failed, here are strategies to optimize it.

## Bin Definition Experiments

### 1. More Aggressive Top Bin (Focus on Winners)

**Current**: `[0.25, 0.5, 0.75]` = 25/25/25/25 split

**Try**: More selective top bin
```yaml
WINNER_BINS_Q: "[0.33, 0.67, 0.85]"  # 33/33/18/15 split - smaller top bin
WINNER_BINS_Q: "[0.4, 0.7, 0.9]"      # 40/30/20/10 split - very selective top
```

**Rationale**: If top-bin predictions are good, make the bar higher to get even better trades.

### 2. Focus on Extremes (Avoid Middle)

**Try**: Larger middle bins
```yaml
WINNER_BINS_Q: "[0.2, 0.5, 0.8]"  # 20/30/30/20 split - emphasize extremes
```

**Trading strategy**: Only trade predicted bin 0 (short) or bin 3 (long), skip middle bins.

### 3. Percentile-Based (vs Quantile)

**Alternative**: Use fixed return thresholds instead of quantiles
```python
# In a09label_data.py
def assign_bins_by_threshold(g: pd.DataFrame) -> pd.DataFrame:
    s = g["return_pct"]

    def to_bin(x):
        if x < -0.02: return 0      # < -2%
        if x < 0: return 1           # -2% to 0%
        if x < 0.02: return 2        # 0% to +2%
        return 3                      # > +2%

    g["y_bin"] = s.apply(to_bin)
    return g
```

**Trade-off**: More interpretable, but fixed thresholds don't adapt to volatility regimes.

## Feature Engineering for 4-Bin

Since 4-bin predicts magnitude/ranking, emphasize different features:

### 1. Relative Features (vs Absolute)

**Add features comparing to daily averages:**
```python
# In feature engineering
df['delta_vs_day_mean'] = df.groupby('trade_date')['delta'].transform(
    lambda x: x - x.mean()
)
df['ivr_vs_day_mean'] = df.groupby('trade_date')['ivRank'].transform(
    lambda x: x - x.mean()
)
df['moneyness_rank_daily'] = df.groupby('trade_date')['moneyness'].rank(pct=True)
```

**Rationale**: If model predicts relative performance, give it relative features.

### 2. Momentum Features

**Add more price momentum signals:**
```python
df['price_momentum_5d_10d'] = df['return_5d'] - df['return_10d']
df['price_acceleration'] = df['return_2d'] - df['return_5d']
df['relative_strength'] = df.groupby('trade_date')['return_5d'].rank(pct=True)
```

### 3. Volatility Regime

**Add regime indicators:**
```python
df['vix_percentile_20d'] = df['VIX'].rolling(20).apply(
    lambda x: (x.iloc[-1] > x).mean()
)
df['high_vol_regime'] = (df['VIX'] > df['VIX'].rolling(60).mean()).astype(int)
```

## Model Tuning for Multi-Class

### 1. Ordinal-Aware Loss

**LightGBM with custom loss** (optional advanced):
```python
# Penalize confusion proportional to distance
# Confusing bin 0 with bin 1 = penalty 1
# Confusing bin 0 with bin 3 = penalty 3

def ordinal_objective(y_true, y_pred):
    # Custom implementation
    pass
```

**Or simpler**: Use predicted probabilities as continuous score
```python
# Inference time
df['expected_bin'] = (
    0 * df['prob_bin0'] +
    1 * df['prob_bin1'] +
    2 * df['prob_bin2'] +
    3 * df['prob_bin3']
)
# Trade top 20% by expected_bin
threshold = df['expected_bin'].quantile(0.8)
trades = df[df['expected_bin'] >= threshold]
```

### 2. Class Weights

**If bins are imbalanced after per-day quantiling:**
```yaml
# For LGBM
LGBM_CLASS_WEIGHT: "balanced"  # Or custom: {0: 1, 1: 1, 2: 1.5, 3: 2}
```

**Emphasize getting bin 3 correct**:
```python
class_weight = {0: 1, 1: 1, 2: 1.5, 3: 2.0}
```

### 3. Evaluation Metric

**Optimize for top-bin recall** instead of overall accuracy:
```python
# Custom metric: Reward correctly identifying bin 3
def top_bin_f1(y_true, y_pred):
    # Binary version: Is it bin 3?
    y_true_bin = (y_true == 3).astype(int)
    y_pred_bin = (y_pred == 3).astype(int)
    return f1_score(y_true_bin, y_pred_bin)
```

## Trading Strategy Optimization

### 1. Probability Thresholds (Better than Hard Class)

**Instead of**: "Trade all predicted bin 3"
```python
trades = df[df['bin_pred'] == 3]
```

**Try**: "Trade high-confidence bin 3"
```python
trades = df[df['prob_bin3'] > 0.4]  # Tune threshold
```

**Or**: Expected value approach
```python
# Compute expected return using predicted distribution
df['expected_return'] = (
    df['bin0_avg_return'] * df['prob_bin0'] +
    df['bin1_avg_return'] * df['prob_bin1'] +
    df['bin2_avg_return'] * df['prob_bin2'] +
    df['bin3_avg_return'] * df['prob_bin3']
)
trades = df.nlargest(50, 'expected_return')  # Top 50 by expected return
```

### 2. Multi-Strategy Portfolio

**Combine multiple filters:**
```python
# Strategy A: Only very confident top bin
strat_a = df[(df['prob_bin3'] > 0.5)]

# Strategy B: Confident top 2 bins
strat_b = df[(df['prob_bin3'] > 0.3) | (df['prob_bin2'] > 0.5)]

# Strategy C: Avoid bottom bin
strat_c = df[df['prob_bin0'] < 0.2]
```

**Test separately and combine**:
```python
# 50% allocation to strategy A, 50% to strategy B
portfolio = pd.concat([
    strat_a.sample(frac=0.5, random_state=42),
    strat_b.sample(frac=0.5, random_state=42)
])
```

### 3. Position Sizing by Confidence

**Weight trades by model confidence:**
```python
df['position_size'] = df['prob_bin3']  # Or (prob_bin3 - 0.25) / 0.75 to normalize

# Top-bin trades with size proportional to confidence
trades = df[df['bin_pred'] == 3].copy()
trades['weighted_return'] = trades['return_pct'] * trades['position_size']
```

## Ensemble Approaches

### 1. Different Bin Modes

**Train multiple models:**
```bash
# Model 1: Per-day bins (current)
WINNER_BINS_MODE=per_day WINNER_OUTPUT_DIR=output/bins_perday python b01...

# Model 2: Global bins
WINNER_BINS_MODE=global WINNER_OUTPUT_DIR=output/bins_global python b01...
```

**Ensemble at inference:**
```python
# Average probabilities
df['prob_bin3_avg'] = (df['prob_bin3_perday'] + df['prob_bin3_global']) / 2

# Vote
df['bin_vote'] = df[['bin_pred_perday', 'bin_pred_global']].mode(axis=1)[0]
```

### 2. Different Time Periods

**Train on different train/val splits:**
```python
# Model 1: Train on orig+a, val on b
# Model 2: Train on orig+a+b, val on c
# Model 3: Train on all

# Ensemble: Weighted average based on val performance
```

## Analysis Tools

### 1. Bin Transition Analysis

**Check if predictions are ordinal:**
```python
# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Off-diagonal distance
distances = []
for i in range(4):
    for j in range(4):
        if cm[i, j] > 0:
            distances.extend([abs(i - j)] * cm[i, j])

print(f"Mean prediction error: {np.mean(distances):.2f} bins")
# Good: ~0.5-0.8 (mostly adjacent bin confusion)
# Bad: >1.2 (random predictions)
```

### 2. Calibration Analysis

**Are probabilities meaningful?**
```python
# For predicted prob_bin3 in [0.4, 0.5], what % are actually bin 3?
bins = pd.cut(df['prob_bin3'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
calibration = df.groupby(bins).apply(
    lambda x: (x['y_bin'] == 3).mean()
)
print(calibration)
```

**If not calibrated**: Use isotonic regression or Platt scaling.

### 3. Feature Importance for Each Class

**Which features distinguish bin 3 from bin 0?**
```python
from sklearn.inspection import permutation_importance

# Train separate binary models
# Bin 3 vs rest
y_bin3 = (y == 3).astype(int)
model_bin3 = LGBMClassifier().fit(X, y_bin3)

importance = permutation_importance(model_bin3, X_val, y_val)
# Top features for identifying best trades
```

## Monitoring & Iteration

### 1. Track These Metrics Over Time

```python
metrics = {
    'overall_accuracy': ...,
    'top_bin_precision': ...,  # Of predicted bin 3, how many are actually bin 3?
    'top_bin_recall': ...,     # Of actual bin 3, how many did we predict?
    'mean_return_pred_bin3': ...,  # Portfolio metric
    'sharpe_pred_bin3': ...,
    'top_bin_coverage': ...,   # How many trades predicted as bin 3?
}
```

### 2. A/B Test Strategies

**Compare variations:**
- Bin definition A vs B
- Feature set A vs B
- Threshold 0.3 vs 0.4 vs 0.5

**On held-out data** (e.g., most recent week).

### 3. Walk-Forward Validation

**Since you have time-series data:**
```python
# Train on orig, test on a
# Train on orig+a, test on b
# Train on orig+a+b, test on c
# etc.

# Check if performance is consistent across time periods
```

## Expected Results

### Good Performance Indicators

✅ **Top-bin precision > 35%**: Better than random (25%)
✅ **Mean return of predicted bin 3 > 2x baseline**
✅ **Confusion mainly in adjacent bins**: Shows ordinal learning
✅ **Top-bin capture rate > 40%**: Finding many of the true best trades
✅ **Sharpe ratio improvement > 30% vs baseline**

### Red Flags

🚩 **Top-bin precision < 28%**: Barely better than random
🚩 **Confusion matrix looks random**: Model not learning
🚩 **Mean return of bin 3 predictions ≈ baseline**: Not selective
🚩 **Probabilities not calibrated**: All predictions near 0.25

## Next Steps

1. **Analyze current results**: What are your metrics showing?
2. **Try bin definition variations**: Start with `[0.33, 0.67, 0.85]`
3. **Add relative features**: Compare to daily means
4. **Use probability thresholds**: Instead of hard bin predictions
5. **Ensemble approaches**: Combine per-day and global models

The key insight is: **If relative performance prediction works, lean into it!** Add more relative features, use within-day comparisons, and treat it as a ranking problem.
