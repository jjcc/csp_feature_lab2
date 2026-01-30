# Config Refactoring Proposal

**Date**: 2026-01-30
**Current Issue**: config.yaml has grown messy with 7 datasets, commented-out sections, manual switching

---

## Current Problems

1. **Manual switching**: Change `<<: *original_config` anchor to switch datasets
2. **Commented code**: 20+ lines in winnerscore section for different datasets
3. **Inconsistent configs**: labeling section uses different config than common
4. **Hard to track**: Which configuration is active for training vs scoring?
5. **No automation**: Have to manually update paths when adding new datasets

---

## Proposed Solution: Profile-Based Config

### Key Concepts

**1. Dataset Registry** - Define each dataset once with metadata
**2. Training Profiles** - Named combinations for incremental training (orig, origa, origab, etc.)
**3. Active Profile** - Single variable to change everything
**4. Computed Paths** - Auto-generated from profile name
**5. Separate Scoring Config** - Independent from training profile

### Structure

```yaml
# ============================================
# ACTIVE CONFIGURATION (Single source of truth)
# ============================================
active:
  train_profile: "origabcde"  # Which datasets to train on
  score_dataset: "f"          # Which dataset to score
  model_version: "v9"         # Model version tag

# ============================================
# DATASET REGISTRY (Define once, use many times)
# ============================================
datasets:
  orig:
    tag: "orig"
    period: "2025-04-25 to 2025-08-11"
    description: "Baseline training period"
    data_dir: "option/put/unprocessed"
    data_basic_csv: "trades_raw_orig.csv"
    cutoff_date: "2025-08-16"

  a:
    tag: "a"
    period: "2025-08-11 to 2025-08-29"
    description: "Post-Aug validation becoming training"
    data_dir: "option/put/put25_0811-0829"
    data_basic_csv: "trades_raw_a_0811.csv"
    cutoff_date: "2025-09-13"

  b:
    tag: "b"
    period: "2025-09-01 to 2025-09-12"
    data_dir: "option/put/put25_0901-0912"
    data_basic_csv: "trades_raw_b_0901.csv"
    cutoff_date: "2025-09-20"

  c:
    tag: "c"
    period: "2025-09-15 to 2025-09-26"
    data_dir: "option/put/put25_0915-0926"
    data_basic_csv: "trades_raw_c_0915.csv"
    cutoff_date: "2025-10-04"

  d:
    tag: "d"
    period: "2025-09-29 to 2025-10-10"
    data_dir: "option/put/put25_0929-1010"
    data_basic_csv: "trades_raw_d_0929.csv"
    cutoff_date: "2025-10-18"

  e:
    tag: "e"
    period: "2025-10-13 to 2025-10-24"
    data_dir: "option/put/put25_1013-1024"
    data_basic_csv: "trades_raw_e_1013.csv"
    cutoff_date: "2025-11-01"

  f:
    tag: "f"
    period: "2025-10-27 to 2025-11-07"
    data_dir: "option/put/put25_1027-1107"
    data_basic_csv: "trades_raw_f_1027.csv"
    cutoff_date: "2025-11-15"

# ============================================
# TRAINING PROFILES (Walk-forward combinations)
# ============================================
profiles:
  orig:
    datasets: ["orig"]
    description: "Baseline only (Apr-Aug)"

  origa:
    datasets: ["orig", "a"]
    description: "Baseline + Aug (validated on a, now training)"

  origab:
    datasets: ["orig", "a", "b"]
    description: "Through Sep 1 (validated on b, now training)"

  origabc:
    datasets: ["orig", "a", "b", "c"]
    description: "Through Sep 15 (validated on c, now training)"

  origabcd:
    datasets: ["orig", "a", "b", "c", "d"]
    description: "Through Sep 29 (validated on d, now training)"

  origabcde:
    datasets: ["orig", "a", "b", "c", "d", "e"]
    description: "Through Oct 13 (validated on e, now training)"

  origabcdef:
    datasets: ["orig", "a", "b", "c", "d", "e", "f"]
    description: "Full dataset through Oct 27"

# ============================================
# PATH TEMPLATES (Computed dynamically)
# ============================================
paths:
  # Training paths (use active.train_profile)
  train_input: "output/data_labeled/labeled_merged_with_gex_macro_{train_profile}.csv"
  train_output: "output/winner_train/{model_version}_oof_{train_profile}"
  model_file: "winner_classifier_model_{train_profile}_lgbm.pkl"

  # Scoring paths (use active.score_dataset and train_profile)
  score_input: "output/data_labeled/labeled_trades_with_gex_macro_{score_dataset}.csv"
  score_model: "output/winner_train/{model_version}_oof_{train_profile}/{model_file}"
  score_output: "output/winner_score/{model_version}_model_{train_profile}"

# ============================================
# COMMON SETTINGS (Unchanged)
# ============================================
common:
  output_dir: "output"
  data_prep: "data_prep"
  labeled: "data_labeled"

data:
  glob: "coveredPut_*.csv"
  target_time: "11:00"
  batch_size: 30

gex:
  base_dir: "gex101"
  target_time: "11:00"
  filter: 0

macro:
  vix_csv: "output/vix_data.csv"
  px_base_dir: "output/price_cache"

# ============================================
# WINNER TRAINING CONFIG
# ============================================
winner:
  # Now computed from active.train_profile
  # Scripts will resolve: paths.train_input with {train_profile} = active.train_profile
  train_epsilon: 0.02
  model_name: "winner_classifier_{model_version}"
  train_target: "return_mon"
  model_type: "lgbm"

  # Training params (unchanged)
  features: ""
  id_cols: "symbol,tradeTime,return_pct,return_mon,return_ann,daysToExpiration"
  random_state: 42
  classifier_n_estimators: 400
  class_weight: "balanced_subsample"
  impute_missing: 1
  use_weights: 1
  weight_alpha: 0.08
  target_precision: "0.88,0.92"
  oof_folds: 5
  time_series: 1

# ============================================
# WINNER SCORING CONFIG
# ============================================
winnerscore:
  # Now computed from active.score_dataset and active.train_profile
  proba_col: "win_proba"
  pred_col: "win_predict"
  threshold: ""
  use_pack_best_f1: 1
  target_precision: 0.90
  auto_calibrate: 1
```

### Usage Examples

**1. Train on orig+a+b+c+d+e:**
```yaml
active:
  train_profile: "origabcde"  # Change this one line
  score_dataset: "f"
  model_version: "v9"
```

**2. Score dataset f with model trained on origabcde:**
```yaml
active:
  train_profile: "origabcde"  # Model to use
  score_dataset: "f"          # Data to score
  model_version: "v9"
```

**3. Add new dataset g:**
```yaml
# Step 1: Add to registry
datasets:
  g:
    tag: "g"
    period: "2025-11-10 to 2025-11-21"
    data_dir: "option/put/put25_1110-1121"
    data_basic_csv: "trades_raw_g_1110.csv"
    cutoff_date: "2025-11-29"

# Step 2: Add profile
profiles:
  origabcdefg:
    datasets: ["orig", "a", "b", "c", "d", "e", "f", "g"]
    description: "Full dataset through Nov 10"

# Step 3: Score g, train on origabcdef
active:
  train_profile: "origabcdef"
  score_dataset: "g"
```

### Helper Script (Python)

Create `scripts/config_helper.py` to resolve paths:

```python
#!/usr/bin/env python3
"""Helper to resolve config paths dynamically."""
import yaml
import sys

def resolve_paths(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    active = cfg['active']
    paths = cfg['paths']

    # Resolve template strings
    resolved = {}
    for key, template in paths.items():
        resolved[key] = template.format(
            train_profile=active['train_profile'],
            score_dataset=active['score_dataset'],
            model_version=active['model_version']
        )

    return resolved

def get_profile_datasets(profile_name, config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    profile = cfg['profiles'][profile_name]
    datasets = []

    for tag in profile['datasets']:
        datasets.append(cfg['datasets'][tag])

    return datasets

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        paths = resolve_paths()
        for key, path in paths.items():
            print(f"{key}: {path}")
    elif len(sys.argv) > 1 and sys.argv[1] == "profile":
        profile = sys.argv[2] if len(sys.argv) > 2 else "origabcde"
        datasets = get_profile_datasets(profile)
        print(f"Profile '{profile}' includes:")
        for ds in datasets:
            print(f"  - {ds['tag']}: {ds['period']}")
```

### Migration Path

**Step 1: Create new config structure** (keep old one as backup)
```bash
cp config.yaml config.yaml.backup
# Create new config with profile structure
```

**Step 2: Update scripts to use new structure**
```python
# In training scripts:
from scripts.config_helper import resolve_paths, get_profile_datasets

paths = resolve_paths()
train_input = paths['train_input']
model_output = paths['train_output']

# Get all datasets in active profile
profile = config['active']['train_profile']
datasets = get_profile_datasets(profile)
```

**Step 3: Test with current profile**
```bash
# Should produce same results as before
python scripts/config_helper.py show
python b01train_winner_classifier_pct_oof.py
```

**Step 4: Gradually migrate scoring scripts**

### Benefits

✅ **Single place to change**: active.train_profile and active.score_dataset
✅ **No commented code**: All configs are named and selectable
✅ **Easy to add datasets**: Just add to registry and create profile
✅ **Self-documenting**: Each profile has description
✅ **Computed paths**: No manual path editing
✅ **Walk-forward clarity**: Profile names show incremental growth
✅ **Scoring independence**: Can score any dataset with any model

### Comparison

| Current | Proposed |
|---------|----------|
| Edit `<<: *config` anchor | Change `active.train_profile` |
| 20+ commented winnerscore lines | Computed from active vars |
| Manual path construction | Auto-computed templates |
| Hard to track active config | Clear active section at top |
| No dataset metadata | Full registry with periods |
| Profiles implicit in filenames | Explicit profiles with descriptions |

---

## Alternative: Simple Cleanup (If full refactor is too much)

If the full profile approach is too big a change, here's a simpler cleanup:

### Minimal Changes

1. **Remove all commented lines** in winnerscore
2. **Use environment variables** for frequently changed values:
   ```yaml
   active:
     train_profile: ${TRAIN_PROFILE:-origabcde}
     score_dataset: ${SCORE_DATASET:-f}
   ```

3. **Computed paths with string substitution**:
   ```yaml
   winnerscore:
     score_input: "output/data_labeled/labeled_trades_with_gex_macro_${SCORE_DATASET}.csv"
     model_in: "output/winner_train/v8_oof_${TRAIN_PROFILE}/winner_classifier_model_${TRAIN_PROFILE}_lgbm.pkl"
   ```

4. **Use .env files** for different scenarios:
   ```bash
   # .env.train_full
   TRAIN_PROFILE=origabcde

   # .env.score_f
   SCORE_DATASET=f
   TRAIN_PROFILE=origabcde
   ```

---

## Recommendation

I recommend the **Profile-Based Configuration** approach because:

1. You already have the dataset structure (orig, a, b, c, d, e, f)
2. You already use walk-forward validation (a07 script)
3. Your naming convention (origa, origab, origabc) naturally maps to profiles
4. One-time refactor effort but long-term maintainability
5. Makes it easy to add new datasets every 2 weeks

The alternative is good for quick cleanup but doesn't solve the fundamental organization issue.

What do you think? Would you like me to help implement either approach?
