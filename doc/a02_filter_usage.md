# a02_filter_noisy_trades.py - Usage Guide

## Purpose

Remove option trades that occur too close to corporate events (earnings, splits) to reduce noise in training data.

## Why Filter Near Events?

Corporate events cause abnormal price volatility and options behavior:
- **Earnings**: IV crush post-announcement, gap moves
- **Stock splits**: Price discontinuities, liquidity changes, options chain adjustments

Trading near these events introduces noise that's hard for models to learn from consistently.

## How It Works

Two-phase filtering:

1. **Trade Phase**: Exclude if `tradeTime` is within exclusion window of any event
2. **Expiry Phase**: Exclude if `expirationDate` is within exclusion window of any event

Both conditions are checked independently - a trade is excluded if EITHER phase triggers.

## Configuration

Edit `corp_action_config.yaml`:

```yaml
# Input/Output
trades_input_csv: "option/put/unprocessed/trades_raw_orig.csv"
filtered_trades_csv: "option/put/filtered/trades_filtered_orig.csv"

# Exclusion windows (days)
exclusion_windows:
  earnings:
    days_before_trade: 2   # Don't open positions 2 days before earnings
    days_after_trade: 1    # Don't open positions 1 day after earnings
    days_before_expiry: 7  # Don't let positions expire 7 days before earnings
    days_after_expiry: 1   # Don't let positions expire 1 day after earnings

  split:
    days_before_trade: 5
    days_after_trade: 3
    days_before_expiry: 10
    days_after_expiry: 5
```

### Tuning Exclusion Windows

**Conservative (wider windows)**:
- Removes more trades
- Cleaner dataset
- Risk: Less training data

**Aggressive (narrower windows)**:
- Keeps more trades
- More training data
- Risk: More noise

**Recommended starting points**:
- Earnings: ±2 days (trade), ±7 days (expiry)
- Splits: ±5 days (trade), ±10 days (expiry)

## Usage

### Basic Usage

```bash
# Run with default config
python a02_filter_noisy_trades.py
```

### Pipeline Integration

```bash
# Step 1: Collect corporate events
python edga_events_scrap.py
# Output: output/data_prep/corp_events/events_apr25_aug08.csv

# Step 2: Filter trades near events
python a02_filter_noisy_trades.py
# Output: option/put/filtered/trades_filtered_orig.csv

# Step 3: Continue with feature engineering
python a03_build_dataset_with_features.py --input option/put/filtered/trades_filtered_orig.csv
```

## Output Files

1. **Filtered trades CSV**: Clean trades ready for feature engineering
2. **Excluded trades CSV** (optional): Trades removed, for analysis
3. **Filter report TXT**: Statistics on what was excluded

### Example Report

```
======================================================================
Corporate Events Trade Filtering Report
======================================================================

Overall Statistics:
  Original trades:        50,000
  Trades kept:            42,500 (85.0%)
  Trades excluded:        7,500 (15.0%)

Exclusion Reasons (by phase):
  Trade phase:            3,200 trades
    - EARNINGS_before_1d                   1,500
    - EARNINGS_before_2d                   1,000
    - SPLIT_before_3d                        700

  Expiry phase:           4,800 trades
    - EARNINGS_before_5d                   2,100
    - EARNINGS_before_6d                   1,500
    - SPLIT_before_8d                      1,200

Exclusions by Event Type:
  EARNINGS:
    Window: ±2/1 days (trade), ±7/1 days (expiry)
    Excluded: 5,200 trades (2,500 trade phase, 3,700 expiry phase)

  SPLIT:
    Window: ±5/3 days (trade), ±10/5 days (expiry)
    Excluded: 2,800 trades (700 trade phase, 2,100 expiry phase)
======================================================================
```

## Algorithm Details

### Event Proximity Detection

Uses pandas `merge_asof` for efficient temporal joins:

```python
# For each trade, find:
- nearest_event_before: Most recent event before trade/expiry date
- nearest_event_after: Next event after trade/expiry date
- days_to_event: Distance in calendar days
```

### Exclusion Logic

```python
for trade in trades:
    # Check trade date proximity
    if event_before exists:
        if 0 <= days_to_event_before <= days_before_trade:
            EXCLUDE (reason: "{EVENT_TYPE}_before_{days}d")

    if event_after exists:
        if 0 <= days_to_event_after <= days_after_trade:
            EXCLUDE (reason: "{EVENT_TYPE}_after_{days}d")

    # Repeat for expiry date...
```

## Common Issues

### Too Many Trades Excluded

**Symptoms**: >30% of trades removed

**Solutions**:
1. Reduce exclusion window sizes
2. Check if corporate events data is correct
3. Consider if your symbols are too event-heavy (e.g., small-cap biotechs with frequent announcements)

### Too Few Trades Excluded

**Symptoms**: <5% of trades removed

**Solutions**:
1. Verify events CSV has data for your date range
2. Check symbol name matching (case sensitivity, special chars)
3. Increase exclusion window sizes

### Performance Issues

**Symptoms**: Script takes >5 minutes for 100k trades

**Solutions**:
1. Ensure input CSVs are reasonably sized (<1M rows)
2. Check for duplicate events (dedupe events CSV first)
3. Use `keep_filtered_trades: false` to skip saving excluded trades

## Integration with Existing Pipeline

### Before a02 (Current State)

```
a00_build_dataset → a09_label_data → b01_train_winner
```

### After a02 (New State)

```
edga_events_scrap (a01) → a02_filter_noisy_trades → a00_build_dataset (rename to a03) → a09_label_data → b01_train_winner
```

### Migration Path

1. Run `edga_events_scrap.py` once to generate events CSV
2. Test `a02_filter_noisy_trades.py` on a small dataset
3. Review filter report - adjust windows if needed
4. Apply to full dataset
5. Compare model performance before/after filtering

## Advanced: Custom Filtering Logic

To add custom event types (dividends, M&A announcements):

1. Add events to unified CSV with appropriate `event_type`
2. Add exclusion window to config:
   ```yaml
   exclusion_windows:
     dividend:
       days_before_trade: 1
       days_after_trade: 0
       days_before_expiry: 3
       days_after_expiry: 0
   ```
3. Run filter - it will automatically handle new event type

## Validation

After filtering, verify results:

```python
import pandas as pd

# Load filtered trades
trades = pd.read_csv("option/put/filtered/trades_filtered_orig.csv")
events = pd.read_csv("output/data_prep/corp_events/events_apr25_aug08.csv")

# Spot check: Pick a known event
nvda_split = events[(events['ticker'] == 'NVDA') & (events['event_type'] == 'SPLIT')]
print(nvda_split['event_date'].iloc[0])  # e.g., 2024-06-10

# Check no NVDA trades near that date
nvda_trades = trades[trades['baseSymbol'] == 'NVDA']
nvda_trades['tradeTime'] = pd.to_datetime(nvda_trades['tradeTime'])
near_split = nvda_trades[
    (nvda_trades['tradeTime'] >= '2024-06-05') &
    (nvda_trades['tradeTime'] <= '2024-06-13')
]
print(f"NVDA trades within ±3 days of split: {len(near_split)}")
# Should be 0 if window >= 3 days
```

## Next Steps

After filtering:
1. Proceed with feature engineering (a03)
2. Label data (a09)
3. Train models (b01)
4. Compare metrics vs unfiltered baseline
5. Iterate on exclusion windows based on model performance
