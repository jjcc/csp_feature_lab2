# 4-Bin Classification Strategy Analysis

## Problem with Binary Classification

After adding corporate event filtering (a02), binary classification performance didn't improve. Possible reasons:

1. **Imbalanced threshold**: Binary win/loss at 0% may not be optimal
   - Many "winners" have tiny positive returns (noise)
   - High performers get same weight as marginal winners

2. **Lost information**: Flattening continuous returns to binary loses signal
   - A +5% return and +0.1% return both = "win"
   - A -5% loss and -0.1% loss both = "loss"

3. **Portfolio construction**: Binary classification doesn't help select BEST trades
   - Knowing a trade will be profitable isn't enough
   - Need to rank trades by expected magnitude

## Why 4-Bin Classification Helps

### 1. Better Signal Preservation

**Quartile bins retain more information:**
```
Binary:  loss | win
         ---- | ----

4-Bin:   Q1  | Q2  | Q3  | Q4
         ---- ---- ---- ----
         worst      best
```

**Model learns:**
- What makes a trade top quartile vs bottom quartile
- Features that drive magnitude, not just direction

### 2. Portfolio Optimization

**Trade selection strategy:**
```python
# Binary: Take all predicted winners
trades = df[df['win_pred'] == 1]

# 4-Bin: Take only top-quartile predictions
trades = df[df['bin_pred'] == 3]  # Or bin_pred >= 2

# Or use predicted probabilities
trades = df[df['prob_bin3'] > 0.4]
```

**Benefits:**
- More selective - only high-conviction trades
- Better risk-adjusted returns
- Natural tiering for position sizing

### 3. Per-Day Normalization

**Your implementation uses per-day quartiles:**
```python
bins_mode = "per_day"  # Compute quartiles within each day
```

**Why this matters:**
- Market regimes vary day-to-day
- A +2% return might be top quartile on a flat day
- Same +2% might be bottom quartile on a volatile day
- Normalizes for intraday volatility

## Implementation Details

### Labeling (a09)

```python
def assign_bins(g: pd.DataFrame) -> pd.DataFrame:
    s = g["return_pct"]
    if s.notna().sum() < 20:
        g["y_bin"] = np.nan
        return g

    q25, q50, q75 = s.quantile([0.25, 0.50, 0.75]).values

    def to_bin(x):
        if x <= q25: return 0  # Worst quartile
        if x <= q50: return 1  # Below median
        if x <= q75: return 2  # Above median
        return 3               # Best quartile

    g["y_bin"] = s.apply(to_bin)
    return g

df = df.groupby("_trade_date", group_keys=False).apply(assign_bins)
```

**Key features:**
- Groups by trade date
- Minimum 20 trades per day to compute stable quartiles
- Assigns NaN if insufficient data (excluded from training)

### Training (b01train_winner_classifier_bins4_oof.py)

**Configuration:**
```bash
WINNER_LABEL_MODE=bins4           # Use 4-bin classification
WINNER_BINS_MODE=per_day          # Quartiles per day
WINNER_BINS_Q=[0.25,0.5,0.75]    # 3 cut points = 4 bins
WINNER_BINS_MIN_GROUP=20          # Min trades per day
WINNER_TIME_COL=captureTime       # Time column for grouping
```

**Model setup:**
- LGBM: `objective="multiclass"`, `num_class=4`
- CatBoost: `loss_function="MultiClass"`
- Metrics: Multi-class log loss, per-class precision/recall

## Evaluation Strategy

### Metrics to Track

1. **Overall multi-class accuracy**
   ```python
   accuracy = (y_pred == y_true).mean()
   ```

2. **Per-class precision/recall**
   ```
   Class 0 (worst): Do we correctly identify bad trades?
   Class 1: Below median
   Class 2: Above median
   Class 3 (best): Do we identify top performers?
   ```

3. **Confusion matrix**
   ```
   How often do we confuse Q1 with Q4?
   Are predictions ordinal (adjacent bins)?
   ```

4. **Top-class capture rate**
   ```python
   # Of actual Q4 trades, how many did we predict as Q3 or Q4?
   capture_rate = df[df['y_true'] == 3]['y_pred'].isin([2, 3]).mean()
   ```

### Portfolio Simulation Metrics

**Compare strategies:**

1. **Baseline**: Trade everything
   ```python
   baseline_return = df['return_pct'].mean()
   ```

2. **Binary classifier**: Trade predicted winners
   ```python
   binary_return = df[df['win_pred'] == 1]['return_pct'].mean()
   ```

3. **Top-bin only**: Trade predicted Q4
   ```python
   top_bin_return = df[df['bin_pred'] == 3]['return_pct'].mean()
   ```

4. **Top-2 bins**: Trade predicted Q3 or Q4
   ```python
   top2_return = df[df['bin_pred'] >= 2]['return_pct'].mean()
   ```

**Key questions:**
- Does top-bin strategy have higher mean return?
- What's the coverage (% of days with signals)?
- Risk-adjusted returns (Sharpe ratio)?

## Recommendations

### 1. Experiment with Bin Definitions

**Try different cut points:**
```bash
# More aggressive - focus on extremes
WINNER_BINS_Q=[0.2,0.5,0.8]  # 20/30/30/20 split

# Isolate top performers
WINNER_BINS_Q=[0.33,0.67,0.9]  # 33/33/24/10 split
```

### 2. Compare Global vs Per-Day

**Run both modes:**
```bash
# Per-day (current)
WINNER_BINS_MODE=per_day

# Global (absolute performance)
WINNER_BINS_MODE=global
```

**Trade-offs:**
- Per-day: Better normalized, more stable predictions
- Global: Absolute performance, might capture regime changes better

### 3. Feature Importance Analysis

**Check which features drive quartile prediction:**
```python
# Do GEX features predict magnitude better than greeks?
# Are macro features more important for multi-class?
```

### 4. Ordinal vs Nominal Classification

**Consider ordinal regression:**
- Current: Treats bins as nominal (unordered classes)
- Alternative: Ordinal regression respects ordering (0 < 1 < 2 < 3)
- May improve predictions by penalizing Q1/Q4 confusion more than Q2/Q3

**Implementation options:**
- LGBM with custom loss function
- Use probabilities: `score = 0*P(0) + 1*P(1) + 2*P(2) + 3*P(3)`
- CatBoost has native ordinal support

### 5. Calibration

**Check if probabilities are calibrated:**
```python
# When model predicts P(class=3) = 0.6, is actual rate ~60%?
from sklearn.calibration import calibration_curve
```

**If not calibrated, use:**
- Platt scaling
- Isotonic regression
- Temperature scaling

## Next Steps

### 1. Run Training

```bash
# Make sure a09 has generated y_bin column
python a09label_data.py

# Train 4-bin classifier
python b01train_winner_classifier_bins4_oof.py
```

### 2. Compare with Binary

**Side-by-side evaluation:**
```bash
# Train binary version
WINNER_LABEL_MODE=binary python b01train_winner_classifier_bins4_oof.py

# Train 4-bin version
WINNER_LABEL_MODE=bins4 python b01train_winner_classifier_bins4_oof.py
```

### 3. Backtesting

**Simulate trading strategies:**
```python
# strategy_comparison.py
strategies = {
    'baseline': lambda df: df,
    'binary_winners': lambda df: df[df['win_pred'] == 1],
    'top_bin': lambda df: df[df['bin_pred'] == 3],
    'top2_bins': lambda df: df[df['bin_pred'] >= 2],
    'high_prob_top': lambda df: df[df['prob_bin3'] > 0.3],
}

for name, filter_fn in strategies.items():
    selected = filter_fn(df)
    metrics = {
        'mean_return': selected['return_pct'].mean(),
        'median_return': selected['return_pct'].median(),
        'coverage': len(selected) / len(df),
        'win_rate': (selected['return_pct'] > 0).mean(),
        'sharpe': selected['return_pct'].mean() / selected['return_pct'].std(),
    }
    print(f"{name}: {metrics}")
```

## Expected Improvements

**If 4-bin classification works well, you should see:**

1. ✅ **Better selectivity**: Top-bin predictions have higher mean returns than binary "win" predictions
2. ✅ **Reduced false positives**: Fewer marginal winners misclassified as top performers
3. ✅ **Actionable tiering**: Natural breakpoints for position sizing (Q4 = 2x size, Q3 = 1x, etc.)
4. ✅ **Better calibration**: Probabilities more meaningful than binary probabilities

**Potential challenges:**

1. ⚠️ **Lower coverage**: Stricter filtering = fewer trades
2. ⚠️ **Harder learning task**: 4-class vs 2-class requires more data
3. ⚠️ **Class imbalance**: Some bins may have fewer examples
4. ⚠️ **Evaluation complexity**: More metrics to track

## Diagnostics if Performance Still Poor

If 4-bin classification doesn't help, check:

1. **Data quality**: Are labels noisy even after corp event filtering?
   ```python
   # Check label distribution
   df.groupby('y_bin')['return_pct'].describe()

   # Check per-day stability
   df.groupby(['_trade_date', 'y_bin']).size().unstack()
   ```

2. **Feature relevance**: Do features actually predict magnitude?
   ```python
   # Simple correlation check
   for feat in features:
       corr = df[[feat, 'return_pct']].corr().iloc[0, 1]
       print(f"{feat}: {corr:.3f}")
   ```

3. **Market regime**: Are returns predictable at all?
   ```python
   # Random forest with default params as sanity check
   from sklearn.ensemble import RandomForestClassifier
   rf = RandomForestClassifier(n_estimators=100, random_state=42)
   rf.fit(X_train, y_train)
   print(f"OOB score: {rf.oob_score_}")  # Should be > 0.3 for 4-class
   ```

4. **Time-based leakage**: Are future features leaking into training?
   ```python
   # Verify trade_date is before expiration_date
   assert (df['trade_date'] <= df['expirationDate']).all()
   ```

5. **Alternative strategies**: Maybe classification isn't the right approach?
   - Regression: Predict return magnitude directly
   - Ranking: Learn to rank trades by expected return
   - Ensemble: Combine binary classifier with regression

---

## Conclusion

Your 4-bin strategy is a solid evolution from binary classification. The per-day normalization is particularly clever for handling varying market conditions. Focus on:

1. Training both binary and 4-bin models for comparison
2. Evaluating on portfolio metrics (mean return, Sharpe) not just accuracy
3. Checking if top-bin predictions actually capture best trades
4. Iterating on bin definitions and feature engineering if needed

The fact that binary performance didn't improve after filtering suggests the problem might be label quality or feature relevance, not just noise from corp events. The 4-bin approach might help by being more robust to label noise in the middle bins.
