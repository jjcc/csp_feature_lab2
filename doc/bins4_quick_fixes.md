# Quick Fixes for 4-Bin Classifier

## Issue 1: Label/Target Mismatch

**Current Problem:**
- Bins created from `return_pct` (in a09label_data.py line 160)
- Model trained on `return_mon` (in config)
- **Result**: Bins don't match what model predicts → -1.2% top-10% return

**Solution: Align bins with training target**

### Option A: Change bins to use return_mon (RECOMMENDED)

Edit `a09label_data.py` line 160:
```python
# Change from:
s = g["return_pct"]

# To:
s = g["return_mon"]  # Or whatever train_target you use
```

Then re-run:
```bash
python a09label_data.py
python b01train_winner_classifier_bins4_oof.py
```

### Option B: Change training target to return_pct

In your .env or config:
```bash
WINNER_TRAIN_TARGET=return_pct  # Instead of return_mon
```

**Why this matters:**
- Currently predicting "which will be top 25% by return_pct"
- But evaluated on return_mon
- If these don't correlate perfectly → confusion

## Issue 2: Bin 0 → Bin 3 Confusion (646 misclassifications)

**Problem:** 30.6% of worst-quartile trades predicted as best.

**Possible causes:**
1. Features that work for top performers don't help identify losers
2. Class imbalance in training
3. Overfitting to noise

### Fix A: Add Features to Separate Extremes

**Add downside risk features:**
```python
# In feature engineering
df['delta_abs'] = df['delta'].abs()  # Extremes more risky
df['moneyness_extreme'] = (df['moneyness'] - 1.0).abs()  # Far OTM/ITM
df['ivr_vol_product'] = df['impliedVolatilityRank1y'] * df['VIX']  # Extreme vol
df['large_position'] = (df['openInterest'] > df['openInterest'].quantile(0.9)).astype(int)
```

### Fix B: Class Weights

**Penalize bin 0/3 confusion more:**
```python
# In b01train_winner_classifier_bins4_oof.py
# For LGBM
class_weight = {
    0: 2.0,  # Emphasize worst trades
    1: 1.0,
    2: 1.0,
    3: 2.0,  # Emphasize best trades
}
```

Or in config:
```bash
LGBM_CLASS_WEIGHT="{0: 2.0, 1: 1.0, 2: 1.0, 3: 2.0}"
```

### Fix C: Two-Stage Classifier

**Stage 1: Binary extremes vs middle**
```python
# Bin 0+3 (extremes) vs Bin 1+2 (middle)
y_stage1 = ((y == 0) | (y == 3)).astype(int)
```

**Stage 2: Within extremes, classify 0 vs 3**
```python
# Only for predicted extremes
mask = (y_stage1_pred == 1)
y_stage2 = y[mask]  # 0 or 3
```

## Issue 3: GEX Features Not Helping

**Feature importance shows GEX at bottom:**
- gex_neg: 9,928
- gex_total_abs: 8,584
- gex_center_abs_strike: 8,214

**Options:**

### Option A: Drop GEX Features
```python
# Remove from feature list
features = BASE_FEATS + NEW_FEATS  # No GEX_FEATS
```

### Option B: Engineer Better GEX Features
```python
# Instead of absolute values, use relative
df['gex_neg_rank'] = df.groupby('trade_date')['gex_neg'].rank(pct=True)
df['gex_vs_oi'] = df['gex_total_abs'] / (df['openInterest'] + 1)
df['gex_skew'] = df['gex_neg'] / (df['gex_total_abs'] + 1e-6)
```

### Option C: Interaction Features
```python
# GEX might matter more in high-vol regimes
df['gex_x_vix'] = df['gex_neg'] * df['VIX']
df['gex_x_ivr'] = df['gex_neg'] * df['impliedVolatilityRank1y']
```

## Quick Wins Checklist

### Priority 1: Fix Label/Target Mismatch
- [ ] Align bins with training target (use return_mon for both)
- [ ] Re-run a09 + b01
- [ ] Check if top-10% return becomes positive
- [ ] Expected impact: **+5-10% top-bin mean return**

### Priority 2: Reduce Bin 0→3 Confusion
- [ ] Add downside risk features
- [ ] Try class weights {0: 2, 1: 1, 2: 1, 3: 2}
- [ ] Expected impact: **Reduce false positives by 20-30%**

### Priority 3: Feature Selection
- [ ] Drop GEX features (or engineer better ones)
- [ ] Add relative/ranking features
- [ ] Expected impact: **+2-3% accuracy, cleaner model**

### Priority 4: Trading Strategy
- [ ] Use probability threshold instead of hard bins
- [ ] Trade only prob(bin3) > 0.5 (instead of all predicted bin 3)
- [ ] Expected impact: **Higher precision, lower coverage**

## Expected Results After Fixes

**Current:**
- Top-bin recall: 71.3%
- Top-10% mean return: -1.2%
- Spread: 42.6%
- Accuracy: 47%

**After Priority 1 fix:**
- Top-bin recall: ~70% (similar)
- Top-10% mean return: **+2-5%** (target-aligned)
- Spread: **40-50%**
- Accuracy: ~47%

**After Priority 1+2:**
- Top-bin recall: ~68-70%
- Top-10% mean return: **+3-6%**
- Spread: **45-55%**
- Accuracy: **49-52%**
- Bin 0→3 errors: **<400** (down from 646)

## Testing the Fixes

### Before/After Comparison

**Metrics to track:**
```python
metrics = {
    # Model metrics
    'accuracy': ...,
    'top_bin_recall': ...,
    'bin0_to_bin3_errors': ...,  # Should decrease

    # Portfolio metrics
    'top10_mean_return': ...,  # Should become positive
    'spread': ...,  # Should increase
    'top_bin_sharpe': ...,

    # Feature metrics
    'top_5_features': ...,  # Should GEX improve?
}
```

### A/B Test Setup

```bash
# Baseline (current)
WINNER_TRAIN_TARGET=return_mon  # But bins use return_pct
python b01... → output/winner_train/v10_baseline

# Fix 1: Aligned target
# Change a09 to use return_mon for bins
python a09... && python b01... → output/winner_train/v10_aligned

# Fix 2: + Class weights
LGBM_CLASS_WEIGHT="{0:2,1:1,2:1,3:2}"
python b01... → output/winner_train/v10_aligned_weighted

# Compare all three
python eval_bins4_vs_binary.py
```

## Next Steps

1. **Immediate**: Fix label/target alignment (Priority 1)
2. **This week**: Add risk features + class weights (Priority 2)
3. **Next week**: Feature engineering iteration (Priority 3)
4. **Ongoing**: Backtest trading strategies (Priority 4)

Focus on Priority 1 first - it's a clear bug that's probably costing 5-10% performance.
