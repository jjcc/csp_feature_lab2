# Unified Config Usage Guide

**Date**: 2026-01-30
**Implementation**: Template-based + unified corp action config
**Status**: ✅ Complete

---

## What Changed

Your pipeline now has **unified configuration** - you only edit `config.yaml` to control the entire pipeline (a01 → a02 → a00 → a09 → b01 → scoring).

---

## The Problem We Solved

**Before**:
- Two separate config files with different conventions
- `config.yaml` for training/scoring (templates)
- `corp_action_config.yaml` for a01/a02 (hardcoded paths)
- Had to manually edit both files to stay in sync

**After**:
- One source of truth: `config.yaml`
- Change 3 variables at the top, entire pipeline auto-updates
- `corp_action_config.yaml` only has common behavior settings

---

## How to Use

### Change Active Configuration (Change 3 lines at top of config.yaml)

```yaml
# At the top of config.yaml
active_train_profile: "origabcde"  # Which datasets to train on
active_score_dataset: "f"          # Which dataset to score
active_process_dataset: "f"        # Which dataset to process (a01/a02)
```

**That's it!** Everything auto-resolves:

| Variable | Used By | Auto-sets |
|----------|---------|-----------|
| `active_train_profile` | b01 training | Training input, output dir, model name |
| `active_score_dataset` | Scoring scripts | Score input, output paths |
| `active_process_dataset` | a01/a02 | Event dates, tickers file, filtered trade paths |

---

## Complete Pipeline Example

### Process Dataset F (New Data)

**Step 1: Edit config.yaml (3 lines)**
```yaml
active_train_profile: "origabcde"   # Train on orig+a+b+c+d+e
active_score_dataset: "f"           # Score new dataset f
active_process_dataset: "f"         # Process new dataset f
```

**Step 2: Run pipeline**
```bash
# Collect corporate events for dataset f
python a01_collect_corp_events.py
# Auto-uses:
#   - Date range: 2025-10-27 to 2025-11-07
#   - Tickers: output/data_prep/corp_events/symbols_in_option_data_f.txt
#   - Output: output/data_prep/corp_events/events_f.csv

# Filter trades near earnings/splits
python a02_filter_noisy_trades.py
# Auto-uses:
#   - Input: option/put/put25_1027-1107/trades_raw_f_1027.csv
#   - Events: output/data_prep/corp_events/events_f.csv
#   - Output: option/put/filtered/trades_filtered_f.csv

# Build features
python a00build_dataset_with_features.py
# Uses filtered trades from a02

# Label outcomes
python a09label_data.py
# Creates: output/data_labeled/labeled_trades_with_gex_macro_f.csv

# Train on orig+a+b+c+d+e
python b01train_winner_classifier_pct_oof.py
# Auto-uses:
#   - Input: output/data_labeled/labeled_merged_with_gex_macro_origabcde.csv
#   - Output: output/winner_train/v9_oof_origabcde/

# Score dataset f with origabcde model
python score_winner_classifier_env.py
# Auto-uses:
#   - Input: output/data_labeled/labeled_trades_with_gex_macro_f.csv
#   - Model: output/winner_train/v8_oof_origabcde/winner_classifier_model_origabcde_lgbm.pkl
#   - Output: output/winner_score/v8_model_origabcde/scores_winner_lgbm_f.csv
```

---

## Dataset Configuration Structure

Each dataset in `common_configs` now has:

```yaml
common_configs:
  oct_27: &oct_27_config
    # Feature engineering settings
    data_dir: "option/put/put25_1027-1107"
    data_basic_csv: "trades_raw_f_1027.csv"
    output_csv: "labeled_trades_f_1027.csv"
    cutoff_date: "2025-11-15"

    # Corp events settings (for a01)
    events_start_date: "2025-10-27"
    events_end_date: "2025-11-07"
    events_output: "output/data_prep/corp_events/events_f.csv"
    tickers_file: "output/data_prep/corp_events/symbols_in_option_data_f.txt"

    # Filtered trades settings (for a02)
    filtered_trades_csv: "option/put/filtered/trades_filtered_f.csv"
    filtered_out_csv: "option/put/filtered/trades_excluded_f.csv"
```

---

## Adding New Dataset (Example: Dataset G)

**Step 1: Add to config.yaml**
```yaml
# Add to common_configs section
nov_10: &nov_10_config
  data_dir: "option/put/put25_1110-1121"
  data_basic_csv: "trades_raw_g_1110.csv"
  output_csv: "labeled_trades_g_1110.csv"
  cutoff_date: "2025-11-29"

  # Corp events settings
  events_start_date: "2025-11-10"
  events_end_date: "2025-11-21"
  events_output: "output/data_prep/corp_events/events_g.csv"
  tickers_file: "output/data_prep/corp_events/symbols_in_option_data_g.txt"

  # Filtered trades settings
  filtered_trades_csv: "option/put/filtered/trades_filtered_g.csv"
  filtered_out_csv: "option/put/filtered/trades_excluded_g.csv"
```

**Step 2: Add mapping in service/env_config.py**
```python
# In get_active_dataset_config() method, add to tag_to_key dict:
tag_to_key = {
    'orig': 'original',
    'a': 'aug_11',
    'b': 'sep_1',
    'c': 'sep_15',
    'd': 'sep_29',
    'e': 'oct_13',
    'f': 'oct_27',
    'g': 'nov_10',  # ← Add this line
}
```

**Step 3: Update active config**
```yaml
active_process_dataset: "g"  # Process new dataset g
```

**Step 4: Run pipeline**
```bash
python a01_collect_corp_events.py  # All paths auto-resolve!
python a02_filter_noisy_trades.py
# ... continue pipeline
```

---

## What Each Config File Does

### config.yaml (Main Config)
- **Active variables**: Control which datasets to use
- **Dataset registry**: All dataset-specific paths and dates
- **Training settings**: Winner classifier parameters
- **Scoring settings**: Threshold calibration

### corp_action_config.yaml (Common Behavior)
- **SEC settings**: User agent, sleep time, cache dir
- **Exclusion windows**: How many days before/after events to filter
- **Column names**: CSV column mappings

**Key insight**: Dataset-specific settings in `config.yaml`, behavior settings in `corp_action_config.yaml`.

---

## Common Workflows

### Workflow 1: Process and Score New Data Every 2 Weeks

```yaml
# Week 0: Process dataset f, train on origabcde, score f
active_train_profile: "origabcde"
active_score_dataset: "f"
active_process_dataset: "f"
```

```bash
# Run full pipeline
python a01_collect_corp_events.py
python a02_filter_noisy_trades.py
python a00build_dataset_with_features.py
python a09label_data.py
python b01train_winner_classifier_pct_oof.py
python score_winner_classifier_env.py
```

```yaml
# Week 2: Process dataset g, train on origabcdef, score g
active_train_profile: "origabcdef"  # ← Include f in training
active_score_dataset: "g"           # ← Score new data
active_process_dataset: "g"         # ← Process new data
```

### Workflow 2: Reprocess Old Dataset with New Settings

```yaml
# Re-collect events for dataset a with new date range
active_process_dataset: "a"
```

Edit the dataset config:
```yaml
aug_11: &aug_11_config
  # ... other fields
  events_start_date: "2025-08-05"  # ← Expanded window
  events_end_date: "2025-08-31"
```

Run:
```bash
python a01_collect_corp_events.py  # Uses new date range
python a02_filter_noisy_trades.py
```

### Workflow 3: Test Different Exclusion Windows

Edit `corp_action_config.yaml`:
```yaml
exclusion_windows:
  earnings:
    days_before_trade: 3   # ← Increased from 2
    days_after_trade: 2    # ← Increased from 1
```

Re-run:
```bash
python a02_filter_noisy_trades.py  # Uses new exclusion windows
```

No need to change `config.yaml` - exclusion windows are behavior settings, not dataset-specific.

---

## Troubleshooting

### Error: "No active dataset configuration found"

**Problem**: `active_process_dataset` not set or invalid value.

**Solution**: Check config.yaml top section:
```yaml
active_process_dataset: "f"  # Must be one of: orig, a, b, c, d, e, f
```

### Error: "Config must include user_agent"

**Problem**: `corp_action_config.yaml` missing or corrupted.

**Solution**: Check `corp_action_config.yaml` has:
```yaml
user_agent: "Your Name your@email.com"
```

### Error: FileNotFoundError for tickers file

**Problem**: Dataset's `tickers_file` doesn't exist yet.

**Solution**: Create the tickers file first:
```bash
# Extract symbols from raw trades
python -c "
import pandas as pd
df = pd.read_csv('option/put/put25_1027-1107/trades_raw_f_1027.csv')
symbols = sorted(df['baseSymbol'].unique())
with open('output/data_prep/corp_events/symbols_in_option_data_f.txt', 'w') as f:
    f.write('\n'.join(symbols))
"
```

### Paths not resolving correctly

**Check template resolution**:
```python
from service.env_config import config
print("Active process dataset:", config.get('active_process_dataset'))
print("Dataset config:", config.get_active_dataset_config())
```

---

## Benefits

✅ **Single source of truth**: Change `active_process_dataset` in one place
✅ **Automatic path sync**: No more manual path editing
✅ **Easy to add datasets**: Just add to `common_configs` and map the tag
✅ **Clean config files**: No commented-out sections
✅ **Walk-forward validation**: Incremental training (orig → origa → origab)
✅ **Independent control**: Process, train, score can use different datasets

---

## Migration Notes

**Backup files created**:
- `config.yaml.backup` - Original config before templates
- `corp_action_config.yaml.backup` - Original corp config before simplification

**To restore old system**:
```bash
cp config.yaml.backup config.yaml
cp corp_action_config.yaml.backup corp_action_config.yaml
```

---

## Next Steps

1. **Test the unified system**:
   ```bash
   # Set active_process_dataset: "f" in config.yaml
   python a01_collect_corp_events.py
   # Verify it uses correct date range and output paths
   ```

2. **Run full pipeline** for current dataset

3. **When next dataset arrives** (in 2 weeks):
   - Add dataset config to `common_configs`
   - Add tag mapping to `service/env_config.py`
   - Change `active_process_dataset`
   - Run pipeline

4. **Tune exclusion windows** if needed:
   - Edit `corp_action_config.yaml` exclusion_windows
   - Re-run a02
   - Compare filtered trade counts

---

**Questions?**

- See `doc/config_quick_fix_guide.md` for training/scoring config details
- See `doc/unified_config_proposal.md` for design rationale
- Check CLAUDE.md for pipeline overview

**Enjoy your unified config! 🎉**
