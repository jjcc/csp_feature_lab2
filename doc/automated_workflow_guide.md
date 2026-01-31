# Automated Workflow Guide

## Overview

The data preparation pipeline (a00→a01→a02) is now fully automated and controlled by a single variable in `config.yaml`.

## Quick Start

### 1. Set Active Dataset

Edit the top of `config.yaml`:

```yaml
active_process_dataset: "f"  # Change to desired dataset: orig, a, b, c, d, e, f
```

### 2. Run Pipeline

```bash
# Step 1: Build dataset and extract symbols (automated)
python a00build_dataset_with_features.py

# Step 2: Collect corporate events using the symbols from step 1
python a01_collect_corp_events.py

# Step 3: Filter trades near corporate events
python a02_filter_noisy_trades.py

# Step 4: Label trades
python a09label_data.py
```

## What Changed

### Before (Manual)

```bash
# Step 1: Build dataset
python a00build_dataset_with_features.py

# Step 2: MANUALLY extract symbols
python -c "
import pandas as pd
df = pd.read_csv('option/put/put25_1027-1107/trades_raw_f_1027.csv')
symbols = sorted(df['baseSymbol'].unique())
with open('output/data_prep/corp_events/symbols_in_option_data_f.txt', 'w') as f:
    f.write('\n'.join(symbols))
"

# Step 3: Collect events
python a01_collect_corp_events.py

# Step 4: Filter trades
python a02_filter_noisy_trades.py
```

### After (Automated)

```bash
# Step 1: Build dataset (automatically extracts symbols)
python a00build_dataset_with_features.py

# Step 2: Collect events (uses symbols from step 1)
python a01_collect_corp_events.py

# Step 3: Filter trades
python a02_filter_noisy_trades.py
```

**Symbol extraction is now automatic!** No manual step needed.

## How It Works

### active_process_dataset Controls Everything

When you set `active_process_dataset: "f"` in config.yaml:

1. **a00** reads the `oct_27` config (maps "f" → "oct_27")
   - Processes: `option/put/put25_1027-1107/coveredPut_*.csv`
   - Writes symbols to: `output/data_prep/corp_events/symbols_in_option_data_f.txt`
   - Writes trades to: `option/put/put25_1027-1107/trades_raw_f_1027.csv`

2. **a01** reads the same config
   - Reads symbols from: `output/data_prep/corp_events/symbols_in_option_data_f.txt`
   - Uses date range: `2025-10-27` to `2025-11-07`
   - Writes events to: `output/data_prep/corp_events/events_f.csv`

3. **a02** reads the same config
   - Reads trades from: `option/put/put25_1027-1107/trades_raw_f_1027.csv`
   - Reads events from: `output/data_prep/corp_events/events_f.csv`
   - Writes filtered trades to: `option/put/filtered/trades_filtered_f.csv`

### Dataset Tag Mapping

| active_process_dataset | Config Key | Data Directory |
|------------------------|------------|----------------|
| `orig` | `original` | `option/put/unprocessed` |
| `a` | `aug_11` | `option/put/put25_0811-0829` |
| `b` | `sep_1` | `option/put/put25_0901-0912` |
| `c` | `sep_15` | `option/put/put25_0915-0926` |
| `d` | `sep_29` | `option/put/put25_0929-1010` |
| `e` | `oct_13` | `option/put/put25_1013-1024` |
| `f` | `oct_27` | `option/put/put25_1027-1107` |

## Batch Processing

To process ALL datasets at once:

```bash
python a00build_all_datasets.py
```

This will:
- Process every dataset in `common_configs` (except 'original')
- Extract symbols for each dataset
- Write logs for each dataset

Useful for bulk regeneration after config changes.

## Adding a New Dataset

1. Add to `common_configs` in `config.yaml`:

```yaml
common_configs:
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

2. Add mapping to `service/env_config.py`:

```python
tag_to_key = {
    'orig': 'original',
    'a': 'aug_11',
    'b': 'sep_1',
    'c': 'sep_15',
    'd': 'sep_29',
    'e': 'oct_13',
    'f': 'oct_27',
    'g': 'nov_10',  # Add this line
}
```

3. Set and run:

```yaml
active_process_dataset: "g"
```

```bash
python a00build_dataset_with_features.py
python a01_collect_corp_events.py
python a02_filter_noisy_trades.py
```

## Troubleshooting

### No symbols written

Check that the dataset config has `tickers_file` set:

```python
from service.env_config import config
cfg = config.get_active_dataset_config()
print(cfg.get('tickers_file'))
```

### Wrong dataset processed

Check `active_process_dataset` value:

```python
from service.env_config import config
print(config._load_yaml_config().get('active_process_dataset'))
```

### Symbol file path issues

Verify the full path is correct:

```bash
ls -la output/data_prep/corp_events/symbols_in_option_data_*.txt
```

## Benefits

✅ **Single control point**: Change one variable to switch datasets
✅ **No manual steps**: Symbol extraction is automatic
✅ **Consistent paths**: All scripts use same dataset config
✅ **Less error-prone**: No copy-paste errors in paths
✅ **Easy to reproduce**: Clear which dataset was processed
