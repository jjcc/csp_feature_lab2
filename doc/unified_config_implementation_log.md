# Unified Config Implementation Log

**Date**: 2026-01-30
**Task**: Unify corp_action_config.yaml with config.yaml template system
**Status**: ✅ Complete

---

## Problem Statement

The pipeline had two separate configuration systems:
1. **config.yaml**: Template-based for training/scoring (a00, a09, b01, scoring)
2. **corp_action_config.yaml**: Hardcoded paths for corp events/filtering (a01, a02)

**User pain point**: "I don't want to edit twice in every change"

---

## Solution Implemented

**Option A (Quick Fix)**: Keep separate configs but unify control
- Add `active_process_dataset` to config.yaml
- Add dataset-specific corp event fields to common_configs
- Simplify corp_action_config.yaml to only have common behavior settings
- Update a01/a02 scripts to read active dataset from config.yaml

**Result**: User only edits config.yaml, both systems stay in sync automatically.

---

## Files Modified

### 1. config.yaml

**Added** `active_process_dataset` variable:
```yaml
# ============================================
# ACTIVE CONFIGURATION (Change these 3 lines only)
# ============================================
active_train_profile: "origabcde"
active_score_dataset: "f"
active_process_dataset: "f"        # NEW: For a01/a02 processing
```

**Added** corp event fields to each dataset in `common_configs`:
```yaml
common_configs:
  oct_27: &oct_27_config
    # Existing fields
    data_dir: "option/put/put25_1027-1107"
    data_basic_csv: "trades_raw_f_1027.csv"
    output_csv: "labeled_trades_f_1027.csv"
    cutoff_date: "2025-11-15"

    # NEW: Corp events settings (for a01)
    events_start_date: "2025-10-27"
    events_end_date: "2025-11-07"
    events_output: "output/data_prep/corp_events/events_f.csv"
    tickers_file: "output/data_prep/corp_events/symbols_in_option_data_f.txt"

    # NEW: Filtered trades settings (for a02)
    filtered_trades_csv: "option/put/filtered/trades_filtered_f.csv"
    filtered_out_csv: "option/put/filtered/trades_excluded_f.csv"
```

**Applied to all datasets**: original, aug_11, sep_1, sep_15, sep_29, oct_13, oct_27

### 2. service/env_config.py

**Enhanced** `_resolve_template()` to handle `active_process_dataset`:
```python
def _resolve_template(self, value):
    """Replace {variable} placeholders in strings with values from config."""
    if not isinstance(value, str):
        return value

    if '{active_train_profile}' in value:
        profile = self._config.get('active_train_profile', '')
        value = value.replace('{active_train_profile}', profile)

    if '{active_score_dataset}' in value:
        dataset = self._config.get('active_score_dataset', '')
        value = value.replace('{active_score_dataset}', dataset)

    # NEW: Support for active_process_dataset
    if '{active_process_dataset}' in value:
        dataset = self._config.get('active_process_dataset', '')
        value = value.replace('{active_process_dataset}', dataset)

    return value
```

**Added** `get_active_dataset_config()` method:
```python
def get_active_dataset_config(self):
    """Get configuration for the active processing dataset.

    Returns the dataset config corresponding to active_process_dataset.
    Used by a01/a02 scripts for corp events and trade filtering.
    """
    yaml_config = self._load_yaml_config()
    active_dataset = yaml_config.get('active_process_dataset', '')

    if not active_dataset:
        return {}

    # Map dataset tag to config key
    tag_to_key = {
        'orig': 'original',
        'a': 'aug_11',
        'b': 'sep_1',
        'c': 'sep_15',
        'd': 'sep_29',
        'e': 'oct_13',
        'f': 'oct_27',
    }

    config_key = tag_to_key.get(active_dataset)
    if not config_key:
        return {}

    common_configs = yaml_config.get('common_configs', {})
    return common_configs.get(config_key, {})
```

### 3. corp_action_config.yaml

**Simplified** to only contain common behavior settings (removed all dataset-specific paths):

**Before** (hardcoded):
```yaml
user_agent: "Jay Chen jchen@apption.com"
tickers_file: "output/data_prep/corp_events/symbols_in_option_data_a.txt"
date_range:
  start: "2025-04-25"
  end: "2025-08-08"
output_csv: "output/data_prep/corp_events/events_apr25_aug08.csv"
trades_input_csv: "option/put/unprocessed/trades_raw_orig.csv"
filtered_trades_csv: "option/put/filtered/trades_filtered_orig.csv"
# ... more hardcoded paths
```

**After** (behavior only):
```yaml
# ============================================
# CORP ACTION CONFIGURATION
# ============================================
# Dataset-specific settings (dates, paths) now come from config.yaml
# This file only contains common behavior settings

# ===== EDGAR Settings =====
user_agent: "Jay Chen jchen@apption.com"
sleep_seconds: 0.3
cache_dir: "output/data_prep/.edgar_cache"
max_8k_fetch_per_ticker: 50
collect_splits: true

# ===== Trade Filtering Configuration (a02) =====
keep_filtered_trades: true

exclusion_windows:
  earnings:
    days_before_trade: 2
    days_after_trade: 1
    days_before_expiry: 7
    days_after_expiry: 1
  split:
    days_before_trade: 5
    days_after_trade: 3
    days_before_expiry: 10
    days_after_expiry: 5

# Column names in trades CSV
symbol_col: "baseSymbol"
trade_date_col: "tradeTime"
expiry_col: "expirationDate"
```

**Backup**: Created `corp_action_config.yaml.backup`

### 4. a01_collect_corp_events.py

**Modified** `main()` function to load dataset config from config.yaml:

**Before**:
```python
def main() -> None:
    cfg = load_config("corp_action_config.yaml")

    # ...
    dr = cfg.get("date_range") or {}
    start = iso_date(dr["start"])
    end = iso_date(dr["end"])
    tickers_file = cfg.get("tickers_file", "tickers.txt")
    out_csv = cfg.get("output_csv", "earnings_8k_item202.csv")
```

**After**:
```python
def main() -> None:
    # Load common settings from corp_action_config.yaml
    cfg = load_config("corp_action_config.yaml")

    # Load dataset-specific settings from config.yaml
    from service.env_config import config as env_config
    dataset_cfg = env_config.get_active_dataset_config()

    if not dataset_cfg:
        raise SystemExit(
            "No active dataset configuration found. "
            "Set active_process_dataset in config.yaml (e.g., 'f', 'orig', etc.)"
        )

    # Get date range and paths from dataset config in config.yaml
    start = iso_date(dataset_cfg.get("events_start_date"))
    end = iso_date(dataset_cfg.get("events_end_date"))
    tickers_file = dataset_cfg.get("tickers_file", "tickers.txt")
    out_csv = dataset_cfg.get("events_output", "earnings_8k_item202.csv")

    # Common settings from corp_action_config.yaml
    sleep_s = float(cfg.get("sleep_seconds", 0.3))
    cache_dir = cfg.get("cache_dir", ".edgar_cache")
```

### 5. a02_filter_noisy_trades.py

**Modified** `load_config()` function:

**Before**:
```python
def load_config(config_path: str = "corp_action_config.yaml") -> FilterConfig:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Parse exclusion windows
    # ...

    return FilterConfig(
        trades_csv=cfg.get("trades_input_csv", ""),
        events_csv=cfg.get("output_csv", ""),
        output_csv=cfg.get("filtered_trades_csv", ""),
        # ...
    )
```

**After**:
```python
def load_config(config_path: str = "corp_action_config.yaml") -> FilterConfig:
    """Load configuration from YAML file and config.yaml"""
    # Load common settings from corp_action_config.yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load dataset-specific settings from config.yaml
    from service.env_config import config as env_config
    dataset_cfg = env_config.get_active_dataset_config()

    if not dataset_cfg:
        raise SystemExit(
            "No active dataset configuration found. "
            "Set active_process_dataset in config.yaml (e.g., 'f', 'orig', etc.)"
        )

    # Parse exclusion windows from corp_action_config.yaml
    # ...

    # Build paths from dataset config
    trades_input = os.path.join(dataset_cfg.get("data_dir", ""), dataset_cfg.get("data_basic_csv", ""))

    return FilterConfig(
        trades_csv=trades_input,
        events_csv=dataset_cfg.get("events_output", ""),
        output_csv=dataset_cfg.get("filtered_trades_csv", ""),
        filtered_csv=dataset_cfg.get("filtered_out_csv", ""),
        # ...
    )
```

---

## Documentation Created

1. **doc/unified_config_usage.md** (341 lines)
   - Complete usage guide for unified configuration
   - Examples of common workflows
   - Adding new datasets
   - Troubleshooting guide

2. **Updated CLAUDE.md**
   - Configuration System section updated
   - Added template resolution explanation
   - Added reference to unified_config_usage.md

3. **This file**: unified_config_implementation_log.md

---

## How It Works

### User Workflow (Simple)

**Change 3 variables at top of config.yaml**:
```yaml
active_train_profile: "origabcde"   # Train on orig+a+b+c+d+e
active_score_dataset: "f"           # Score dataset f
active_process_dataset: "f"         # Process dataset f (a01/a02)
```

**Run pipeline**:
```bash
python a01_collect_corp_events.py  # Uses dataset f's date range and paths
python a02_filter_noisy_trades.py  # Uses dataset f's paths
python a00build_dataset_with_features.py
python a09label_data.py
python b01train_winner_classifier_pct_oof.py  # Trains on origabcde
python score_winner_classifier_env.py  # Scores f with origabcde model
```

**All paths auto-resolve!**

### Technical Flow

1. **a01/a02 scripts** import `service.env_config.config`
2. Call `config.get_active_dataset_config()` to get active dataset
3. **env_config** reads `active_process_dataset` from config.yaml
4. Maps dataset tag (e.g., "f") to config key (e.g., "oct_27")
5. Returns the full dataset config from `common_configs.oct_27`
6. Scripts use paths directly from dataset config

**Key insight**: Dataset-specific settings flow from config.yaml, behavior settings from corp_action_config.yaml.

---

## Benefits

✅ **Single source of truth**: Only edit config.yaml
✅ **No manual path editing**: All paths follow naming convention
✅ **Easy to add datasets**: Add to common_configs + tag mapping
✅ **Independent control**: Process, train, score use different datasets
✅ **Clean configs**: No commented-out sections
✅ **Automatic sync**: Change active_process_dataset, all paths update

---

## Testing Checklist

- [x] Added active_process_dataset to config.yaml
- [x] Added corp event fields to all 7 datasets
- [x] Enhanced env_config.py with template resolution
- [x] Added get_active_dataset_config() method
- [x] Simplified corp_action_config.yaml
- [x] Updated a01_collect_corp_events.py
- [x] Updated a02_filter_noisy_trades.py
- [x] Created comprehensive documentation
- [x] Updated CLAUDE.md
- [ ] User testing (next step)

---

## Future Enhancements

If needed, could add:
1. **Full unification**: Merge corp_action_config.yaml into config.yaml completely
2. **More templates**: Add {model_version} for easier model versioning
3. **Profile validation**: Check that profile names match naming convention
4. **Auto-generate profiles**: From dataset tags (orig+a+b → "origab")

---

## Migration Notes

**Backward compatibility**:
- Original config.yaml backed up as `config.yaml.backup`
- Original corp_action_config.yaml backed up as `corp_action_config.yaml.backup`

**To restore**:
```bash
cp config.yaml.backup config.yaml
cp corp_action_config.yaml.backup corp_action_config.yaml
```

---

## Success Criteria Met

✅ User only needs to edit config.yaml (not both files)
✅ a01/a02 read from unified configuration
✅ Naming convention enforced across pipeline
✅ Clean, maintainable configuration system
✅ Documentation complete

---

## User Instructions

**Next steps**:
1. Review `doc/unified_config_usage.md` for usage examples
2. Test the system:
   ```bash
   # Set active_process_dataset: "f" in config.yaml
   python a01_collect_corp_events.py
   # Verify it uses correct paths
   ```
3. When adding dataset g in 2 weeks:
   - Add to common_configs in config.yaml
   - Add 'g': 'nov_10' mapping in service/env_config.py
   - Change active_process_dataset: "g"
   - Run pipeline

**Questions**: See `doc/unified_config_usage.md` troubleshooting section

---

**Implementation complete! 🎉**
