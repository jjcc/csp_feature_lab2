# Config Quick Fix - Usage Guide

**Date**: 2026-01-30
**Implementation**: Template-based naming convention
**Status**: ✅ Complete

---

## What Changed

Your config.yaml now uses **naming convention templates** instead of hardcoded paths. All you need to change are **2 variables at the top**.

---

## How to Use

### 1. Change Training Profile (Every 2 weeks when adding new data)

**To train on orig+a+b+c+d+e:**
```yaml
# At the top of config.yaml
active_train_profile: "origabcde"  # ← Change this line
active_score_dataset: "f"
```

**Automatically sets:**
- Training input: `output/data_labeled/labeled_merged_with_gex_macro_origabcde.csv`
- Training output: `output/winner_train/v9_oof_origabcde/`
- Model file: `winner_classifier_model_origabcde_lgbm.pkl`

### 2. Change Score Dataset

**To score dataset f:**
```yaml
active_train_profile: "origabcde"  # Model trained on orig+a+b+c+d+e
active_score_dataset: "f"          # ← Change this to score f
```

**Automatically sets:**
- Score input: `output/data_labeled/labeled_trades_with_gex_macro_f.csv`
- Model path: `output/winner_train/v8_oof_origabcde/winner_classifier_model_origabcde_lgbm.pkl`
- Score output folder: `output/winner_score/v8_model_origabcde/`
- Score output file: `scores_winner_lgbm_f.csv`

---

## Adding New Dataset (Example: Dataset G)

**When you get new data in 2 weeks:**

### Step 1: Add to common_configs (config.yaml)
```yaml
# Add after oct_27 config:
oct_41: &oct_41_config
  data_dir: "option/put/put25_1110-1121"
  data_basic_csv: "trades_raw_g_1110.csv"
  output_csv: "labeled_trades_g_1110.csv"
  cutoff_date: "2025-11-29"
```

### Step 2: Run data pipeline
```bash
# Collect events (if needed)
python a01_collect_corp_events.py

# Filter trades
python a02_filter_noisy_trades.py

# Build features for dataset g
python a00build_dataset_with_features.py

# Label dataset g
python a09label_data.py

# Create merged dataset origabcdef using a07
python a07merge_dataset_with_features.py
# This creates: labeled_merged_with_gex_macro_origabcdef.csv
```

### Step 3: Update active config
```yaml
# Score g using model trained on origabcdef
active_train_profile: "origabcdef"  # ← Updated
active_score_dataset: "g"           # ← Updated
```

### Step 4: Run workflow
```bash
# Train on origabcdef
python b01train_winner_classifier_pct_oof.py

# Score dataset g
python score_winner_classifier_env.py
```

**That's it!** Only changed 2 lines in config.yaml.

---

## Common Workflows

### Workflow 1: Walk-Forward Validation (Standard)

**Initial: Train on orig, test on a**
```yaml
active_train_profile: "orig"
active_score_dataset: "a"
```

**Week 2: If a validated well, train on orig+a, test on b**
```yaml
active_train_profile: "origa"
active_score_dataset: "b"
```

**Week 4: If b validated well, train on orig+a+b, test on c**
```yaml
active_train_profile: "origab"
active_score_dataset: "c"
```

**Continue pattern...**

### Workflow 2: Re-score Old Dataset with New Model

**Trained new model on origabcdef, want to re-score dataset d:**
```yaml
active_train_profile: "origabcdef"  # Use new model
active_score_dataset: "d"           # Re-score old data
```

### Workflow 3: Compare Different Models

**Compare model trained on orig vs origa on dataset b:**

```bash
# Test 1: Model trained on orig only
# Edit config.yaml:
active_train_profile: "orig"
active_score_dataset: "b"
python score_winner_classifier_env.py

# Test 2: Model trained on orig+a
# Edit config.yaml:
active_train_profile: "origa"
active_score_dataset: "b"
python score_winner_classifier_env.py

# Compare results
```

---

## What Gets Auto-Computed

### Training Paths
```yaml
# When active_train_profile = "origabcde"
winner.input:      "output/data_labeled/labeled_merged_with_gex_macro_origabcde.csv"
winner.output_dir: "output/winner_train/v9_oof_origabcde"
```

### Scoring Paths
```yaml
# When active_train_profile = "origabcde" and active_score_dataset = "f"
winnerscore.score_input:      "output/data_labeled/labeled_trades_with_gex_macro_f.csv"
winnerscore.model_in:         "output/winner_train/v8_oof_origabcde/winner_classifier_model_origabcde_lgbm.pkl"
winnerscore.score_out_folder: "output/winner_score/v8_model_origabcde"
winnerscore.score_out:        "scores_winner_lgbm_f.csv"
```

---

## Naming Convention Rules

**Profile names must follow pattern:**
- Single dataset: `orig`, `a`, `b`, `c`, `d`, `e`, `f`
- Incremental combinations: `origa`, `origab`, `origabc`, `origabcd`, `origabcde`, `origabcdef`

**File naming must match:**
- Merged files: `labeled_merged_with_gex_macro_{profile}.csv`
- Single dataset files: `labeled_trades_with_gex_macro_{dataset}.csv`
- Model files: `winner_classifier_model_{profile}_lgbm.pkl`

**The a07 script enforces this naming when creating merged datasets.**

---

## Benefits

✅ **Change 2 lines** instead of 6+ scattered lines
✅ **No commented code** (clean config file)
✅ **Automatic path sync** (no mistakes)
✅ **Easy to add new datasets** (3 steps)
✅ **Self-documenting** (profile name shows what's included)
✅ **No script changes needed**

---

## Troubleshooting

### Problem: Paths don't resolve correctly

**Check:**
```python
# In Python:
from service.env_config import config
print(config.get('winner.input'))
# Should show: output/data_labeled/labeled_merged_with_gex_macro_origabcde.csv
```

**If it shows `{active_train_profile}` literally:**
- service/env_config.py wasn't updated correctly
- Verify `_resolve_template()` function exists

### Problem: File not found error

**Example error:** `FileNotFoundError: output/data_labeled/labeled_merged_with_gex_macro_origabcde.csv`

**Solution:** Run a07 to create the merged file:
```bash
python a07merge_dataset_with_features.py
```

This creates incremental merged datasets for all profiles.

### Problem: Old commented lines breaking things

**If you still see old commented lines causing issues:**
```bash
# Restore from backup and redo
cp config.yaml.backup config.yaml
# Follow the quick fix guide again
```

---

## Comparison: Before vs After

### Before (Messy)
```yaml
# Have to change multiple lines and uncomment/comment:
winner:
  #input: "output/data_labeled/labeled_merged_with_gex_macro_orig.csv"
  #input: "output/data_labeled/labeled_merged_with_gex_macro_origa.csv"
  input: "output/data_labeled/labeled_merged_with_gex_macro_origabcde.csv"  # ← Change here
  #output_dir: "output/winner_train/v8_oof_orig"
  #output_dir: "output/winner_train/v8_oof_origa"
  output_dir: "output/winner_train/v8_oof_origabcde"  # ← Change here

winnerscore:
  #score_input: "output/data_labeled/labeled_trades_with_gex_macro_a.csv"
  #score_input: "output/data_labeled/labeled_trades_with_gex_macro_b.csv"
  score_input: "output/data_labeled/labeled_trades_with_gex_macro_f.csv"  # ← Change here
  #model_in: "output/winner_train/v8_oof_orig/winner_classifier_model_orig_lgbm.pkl"
  model_in: "output/winner_train/v8_oof_origabcde/winner_classifier_model_origabcde_lgbm.pkl"  # ← Change here
  # ... 20+ more commented lines
```

### After (Clean)
```yaml
# Change 2 lines at the top:
active_train_profile: "origabcde"  # ← Change here
active_score_dataset: "f"          # ← Change here

# Everything else auto-computed:
winner:
  input: "output/data_labeled/labeled_merged_with_gex_macro_{active_train_profile}.csv"
  output_dir: "output/winner_train/v9_oof_{active_train_profile}"

winnerscore:
  score_input: "output/data_labeled/labeled_trades_with_gex_macro_{active_score_dataset}.csv"
  model_in: "output/winner_train/v8_oof_{active_train_profile}/winner_classifier_model_{active_train_profile}_lgbm.pkl"
```

---

## Next Steps

1. **Test the changes:**
   ```bash
   # Verify paths resolve correctly (if you have dependencies installed)
   python -c "from service.env_config import config; print('Input:', config.get('winner.input'))"
   ```

2. **Run your normal workflow:**
   ```bash
   # Train
   python b01train_winner_classifier_pct_oof.py

   # Score
   python score_winner_classifier_env.py
   ```

3. **When adding next dataset (in 2 weeks):**
   - Add to common_configs
   - Run pipeline (a01→a02→a00→a09→a07)
   - Change 2 lines in config.yaml
   - Train and score

---

## Questions?

**Q: Can I still use the old config?**
A: Yes, backup is saved as `config.yaml.backup`

**Q: Do I need to update production script (task_score_tail_winner.py)?**
A: No! It has hardcoded paths and doesn't use config.yaml

**Q: Can I add more template variables?**
A: Yes, edit `_resolve_template()` in `service/env_config.py`

**Q: What if I want different model versions (v8 vs v9)?**
A: The version is still hardcoded in paths (v9_oof for training, v8_oof for scoring). Can make this a variable too if needed.

---

**That's it! Enjoy your cleaner config!** 🎉
