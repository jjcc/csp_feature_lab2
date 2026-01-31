# Unified Config Proposal - Complete Pipeline

**Problem**: Pipeline has two config files with different conventions
- `corp_action_config.yaml` for a01/a02 (hardcoded)
- `config.yaml` for a00/a09/b01 (now template-based)

**Solution**: Merge corp action settings into main config.yaml

---

## Proposed Structure

```yaml
# ============================================
# ACTIVE CONFIGURATION
# ============================================
active_train_profile: "origabcde"  # For training
active_score_dataset: "f"          # For scoring
active_process_dataset: "f"        # For a01/a02 processing (NEW)

# ============================================
# DATASET REGISTRY (Enhanced with corp event settings)
# ============================================
common_configs:
  original: &original_config
    # Existing fields
    data_dir: "option/put/unprocessed"
    data_basic_csv: "trades_raw_orig.csv"
    output_csv: "labeled_trades_with_gex_macro_orig.csv"
    cutoff_date: "2025-08-16"

    # NEW: Corp events settings for a01
    events_start_date: "2025-04-25"
    events_end_date: "2025-08-11"
    events_output: "output/data_prep/corp_events/events_orig.csv"
    tickers_file: "output/data_prep/corp_events/symbols_in_option_data_orig.txt"

    # NEW: Filtered trades settings for a02
    filtered_trades_csv: "option/put/filtered/trades_filtered_orig.csv"
    filtered_out_csv: "option/put/filtered/trades_excluded_orig.csv"

  oct_27: &oct_27_config
    # Existing fields
    data_dir: "option/put/put25_1027-1107"
    data_basic_csv: "trades_raw_f_1027.csv"
    output_csv: "labeled_trades_f_1027.csv"
    cutoff_date: "2025-11-15"

    # NEW: Corp events settings
    events_start_date: "2025-10-27"
    events_end_date: "2025-11-07"
    events_output: "output/data_prep/corp_events/events_f.csv"
    tickers_file: "output/data_prep/corp_events/symbols_in_option_data_f.txt"

    # NEW: Filtered trades settings
    filtered_trades_csv: "option/put/filtered/trades_filtered_f.csv"
    filtered_out_csv: "option/put/filtered/trades_excluded_f.csv"

# ============================================
# CORP ACTION PROCESSING (NEW SECTION)
# ============================================
corp_events:
  # User agent for SEC
  user_agent: "Jay Chen jchen@apption.com"

  # Behavior settings (same for all datasets)
  sleep_seconds: 0.3
  cache_dir: "output/data_prep/.edgar_cache"
  max_8k_fetch_per_ticker: 50
  collect_splits: true

  # Dataset-specific settings come from active_process_dataset
  # These will be resolved from common_configs[active_process_dataset]
  tickers_file: "{tickers_file}"         # Resolved from active dataset config
  date_range:
    start: "{events_start_date}"         # Resolved from active dataset config
    end: "{events_end_date}"             # Resolved from active dataset config
  output_csv: "{events_output}"          # Resolved from active dataset config

trade_filtering:
  # Input/output resolved from active dataset config
  trades_input_csv: "{data_dir}/{data_basic_csv}"
  filtered_trades_csv: "{filtered_trades_csv}"
  filtered_out_csv: "{filtered_out_csv}"
  keep_filtered_trades: true

  # Exclusion windows (same for all datasets)
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

  # Column names
  symbol_col: "baseSymbol"
  trade_date_col: "tradeTime"
  expiry_col: "expirationDate"
```

---

## Enhanced Template Resolution

Update `service/env_config.py` to handle dataset-specific templates:

```python
def _resolve_template(self, value):
    """Replace {variable} placeholders in strings with values from config."""
    if not isinstance(value, str):
        return value

    # Get active profiles
    train_profile = self._config.get('active_train_profile', '')
    score_dataset = self._config.get('active_score_dataset', '')
    process_dataset = self._config.get('active_process_dataset', '')

    # Resolve simple templates
    value = value.replace('{active_train_profile}', train_profile)
    value = value.replace('{active_score_dataset}', score_dataset)
    value = value.replace('{active_process_dataset}', process_dataset)

    # Resolve dataset-specific fields (for a01/a02)
    if process_dataset:
        # Find the config for this dataset
        dataset_config_key = self._find_dataset_config_key(process_dataset)
        if dataset_config_key:
            # Get all fields from that dataset config
            prefix = f"common_configs.{dataset_config_key}."
            for key in self._config:
                if key.startswith(prefix):
                    field = key[len(prefix):]
                    field_value = self._config[key]
                    placeholder = '{' + field + '}'
                    if placeholder in value:
                        value = value.replace(placeholder, str(field_value))

    return value

def _find_dataset_config_key(self, tag):
    """Find which config key (original, aug_11, oct_27, etc.) corresponds to a tag."""
    # Map: orig → original, a → aug_11, f → oct_27, etc.
    # This mapping could be in config or derived from output_csv filenames
    # For simplicity, check which config has the matching tag in its basic_csv
    for key in self._config:
        if key.startswith('common_configs.') and '.data_basic_csv' in key:
            csv_name = self._config[key]
            # Extract tag from csv name (e.g., trades_raw_f_1027.csv → f)
            if f'_raw_{tag}_' in csv_name or f'_raw_{tag}.' in csv_name:
                config_key = key.split('.')[1]
                return config_key

    # Fallback: check if tag matches common pattern
    tag_to_key = {
        'orig': 'original',
        'a': 'aug_11',
        'b': 'sep_1',
        'c': 'sep_15',
        'd': 'sep_29',
        'e': 'oct_13',
        'f': 'oct_27',
    }
    return tag_to_key.get(tag)
```

---

## Usage Example

### Process Dataset F (a01 → a02 → a00 → a09)

```yaml
# Top of config.yaml - change these 3 lines:
active_train_profile: "origabcde"   # For later training
active_score_dataset: "f"           # For later scoring
active_process_dataset: "f"         # For a01/a02 NOW
```

**Run pipeline:**
```bash
# Step 1: Collect events for dataset f
python a01_collect_corp_events.py
# Reads: active_process_dataset = "f"
# Uses: oct_27 config (events_start_date: 2025-10-27, events_end_date: 2025-11-07)
# Outputs: output/data_prep/corp_events/events_f.csv

# Step 2: Filter trades for dataset f
python a02_filter_noisy_trades.py
# Reads: active_process_dataset = "f"
# Input: option/put/put25_1027-1107/trades_raw_f_1027.csv
# Output: option/put/filtered/trades_filtered_f.csv

# Step 3: Build features
python a00build_dataset_with_features.py
# Uses filtered trades from step 2

# Step 4: Label
python a09label_data.py
```

### Add New Dataset G (2 weeks later)

```yaml
# Step 1: Add to config
nov_10: &nov_10_config
  data_dir: "option/put/put25_1110-1121"
  data_basic_csv: "trades_raw_g_1110.csv"
  output_csv: "labeled_trades_g_1110.csv"
  cutoff_date: "2025-11-29"

  # Corp events
  events_start_date: "2025-11-10"
  events_end_date: "2025-11-21"
  events_output: "output/data_prep/corp_events/events_g.csv"
  tickers_file: "output/data_prep/corp_events/symbols_in_option_data_g.txt"

  # Filtered trades
  filtered_trades_csv: "option/put/filtered/trades_filtered_g.csv"
  filtered_out_csv: "option/put/filtered/trades_excluded_g.csv"

# Step 2: Change active config
active_process_dataset: "g"  # ← Process dataset g
```

**Run pipeline** - all paths auto-resolve!

---

## Benefits

✅ **One config file** for entire pipeline (a01 → a02 → a00 → a09 → b01)
✅ **Change 1 line** to process new dataset (`active_process_dataset: "g"`)
✅ **Date ranges per dataset** (stored in dataset config)
✅ **Naming convention enforced** across all stages
✅ **No manual path editing** anywhere
✅ **Can process, train, score different datasets** independently

---

## Migration Steps

1. **Add fields to existing dataset configs** in config.yaml
   - events_start_date, events_end_date
   - events_output, tickers_file
   - filtered_trades_csv, filtered_out_csv

2. **Add new sections** to config.yaml
   - corp_events (settings from corp_action_config.yaml)
   - trade_filtering (settings from corp_action_config.yaml)

3. **Enhance env_config.py**
   - Add active_process_dataset support
   - Enhance _resolve_template() to handle dataset-specific fields

4. **Update a01/a02 scripts**
   - Read from config.yaml instead of corp_action_config.yaml
   - Use resolved templates

5. **Keep corp_action_config.yaml as backup**
   - Rename to corp_action_config.yaml.old

---

## Alternative: Keep Separate (Simpler Migration)

If merging is too complex, keep `corp_action_config.yaml` but add template support:

```yaml
# corp_action_config.yaml with templates
active_dataset: "f"  # Which dataset to process

# Dataset-specific settings
datasets:
  f:
    dates: { start: "2025-10-27", end: "2025-11-07" }
    input: "option/put/put25_1027-1107/trades_raw_f_1027.csv"
    output: "option/put/filtered/trades_filtered_f.csv"

  g:
    dates: { start: "2025-11-10", end: "2025-11-21" }
    input: "option/put/put25_1110-1121/trades_raw_g_1110.csv"
    output: "option/put/filtered/trades_filtered_g.csv"

# Common settings
user_agent: "Jay Chen jchen@apption.com"
# ... rest unchanged
```

Then scripts use `active_dataset` to look up settings.

---

## Recommendation

**For now:** Use the simpler "keep separate" approach
- Less disruption
- Quick to implement
- Can merge later if needed

**Long term:** Merge into config.yaml
- Single source of truth
- Better maintainability
- Enforces naming convention everywhere

What do you think? Want to do the quick separate-config fix or go for full unification?
