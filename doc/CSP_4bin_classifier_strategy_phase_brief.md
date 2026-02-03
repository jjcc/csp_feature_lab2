# Winner Classifier Bins4 OOF (CSP Strategy)

## 1) Goal

Build a **cross-sectional ranker** for CSP trades using existing features, because global win/loss classification plateaued (ROC ~0.53). Switch to **4-bin ordinal labels** (per-day quantiles) and evaluate whether the model can rank “best trades” vs “worst trades” within each decision set/day.

---

## 2) Key conclusions so far

* Removing earnings/split-proxy trades (~7% of rows) did **not** materially improve binary classification; bottleneck was **feature separability**, not noise/imbalance.
* Raising win threshold to balance classes improved little → confirms separability bottleneck.
* Switching to **4-bin per-day quantile labels** produced meaningful skill:

  * Latest metrics: **Accuracy ~0.4691**, **Macro-F1 ~0.4620**
  * Ranking spread (trading-relevant): **Top 10% mean `return_mon` ≈ -1.20**, **Bottom 10% ≈ -43.77**, **Spread ≈ +42.57**
  * Important: Top 10% still slightly negative because **per-day bins always create “best of the day” even on bad days** → next bottleneck is **tradeability gating / regime filtering**.

---

## 3) Data + targets

* Dataset: `/mnt/data/labeled_filtered_trades_with_gex_macro_orig.csv`
* Continuous outcome columns (examples in pipeline):

  * `return_mon` (dollar PnL)
  * `return_pct` (pct of collateral, computed as total_pnl / (strike*100) * 100)
* Original binary label: `won = 1 if return > 0 else 0` (too imbalanced ~77.5%)
* New label: `y_bin ∈ {0,1,2,3}` based on **per-day quartiles** of target (recommend: bin on `return_mon` or `return_pct`, but keep consistent with training target)

---

## 4) The critical bug that was fixed

### Problem

With `TimeSeriesSplit`, early rows never appear in validation → `fold_idx = -1`.
Old logic filled missing OOF probabilities using a scalar median (for multiclass), producing invalid probability vectors that **don’t sum to 1** and pollute evaluation.

### Fix

* **Do NOT fill multiclass NaN proba with a scalar**
* Evaluate metrics/spread using only `valid_oof = (fold_idx != -1)`
* Optionally fill missing rows with uniform `1/num_class`, but never use them for metrics.

This materially improved bins4 results.

---

## 5) Files involved

Uploaded / referenced in this work:

* Training script (original): `b01train_winner_classifier_pct_oof.py`
* Labeling helper: `a09label_data.py`
* Output metrics JSON: `winner_classifier_metrics.json`
* OOF predictions: `winner_scores_oof.csv` (should contain `p_bin0..p_bin3`, `fold`, etc.)

Note: `winner_scores_oof.csv` **does** include `p_bin0..p_bin3`. Rows with `fold = -1` may have default/unscored probabilities depending on fill policy; evaluation should exclude them.

---

## 6) Bins4 labeling rules (recommended)

* **bins_mode = per_day**
* group by normalized day of `captureTime` (or another consistent decision timestamp)
* compute quantiles `[0.25, 0.5, 0.75]` of target inside each day
* assign:

  * bin0: ≤ q25
  * bin1: (q25, q50]
  * bin2: (q50, q75]
  * bin3: > q75
* if day group size < `min_group` (e.g., 20), label as NA and drop

---

## 7) LightGBM multiclass gotcha (already hit once)

If using LightGBM:

* When `objective="multiclass"` and `num_class=4`
* Metric **must** be multiclass compatible:

  * use `"multi_logloss"` (recommended)
* Avoid `"auc"` which triggers: “Multiclass objective and metrics don’t match”

---

## 8) Evaluation that matters (not just accuracy)

For bins4, core measures:

* `accuracy`, `macro_f1`, confusion matrix (expect middle bins to be messy)
* **Ranking spread using `p_bin3`**:

  * Take OOF `p_bin3` score
  * Compare `mean(target)` of top 10% scored vs bottom 10% scored
  * Use only `valid_oof` rows

Current: big positive spread (~+42.6) but top decile mean still slightly negative → need gating.

---

## 9) Next steps (high ROI)

### A) Add “tradeability gate” (absolute)

Because per-day ranking can pick “best of a bad day”, add a first-stage filter:

* Predict whether a day/regime is tradeable (absolute expectancy)
* Use features like `VIX`, `impliedVolatilityRank1y`, `gex_*`, `ret_2d_norm`, `ret_5d_norm`, DTE buckets
  Then:
* Only rank within days that pass gate

### B) Hybrid label: rank + absolute floor

Keep bins4 labels, but only treat bin3 as “actionable” when:

* `p_bin3` high AND
* predicted expected bin value or simple rule meets a floor (e.g. VIX regime, GEX regime)

### C) Prefer “top-vs-rest” simplification

Given confusion matrix: extremes are learnable; middle bins noisy. Consider:

* binary label `top_bin = (y_bin == 3)` for selection

---

## 10) Repro / checklist in new session

1. Load training script and ensure:

   * multiclass objective/metric match (`multi_logloss`)
   * OOF proba shape is `(N,4)`
2. Ensure OOF evaluation filters `fold != -1`
3. Confirm saved OOF file includes:

   * `p_bin0,p_bin1,p_bin2,p_bin3`
   * `y_true,y_pred,fold`
4. Compute:

   * accuracy / macro-F1 on valid rows
   * top10 vs bottom10 spread on valid rows using `p_bin3`
5. If top10 mean target still negative → implement gating.

