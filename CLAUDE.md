# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **Cash-Secured Put (CSP) Option Trading ML Pipeline** that:
1. Ingests option trade snapshots from CSVs
2. Labels outcomes using historical price data (yfinance)
3. Enriches with GEX (gamma exposure), VIX, and macro features
4. Trains ML models to predict profitable trades (Winner Classifier) and avoid tail losses (Tail Loss Classifier)
5. Scores new candidates and runs portfolio simulations

The project uses a hybrid configuration system: `config.yaml` for main settings plus `.env` files for environment-specific overrides.

## Key Commands

### Installation
```bash
pip install -r requirements.txt
```

### Data Pipeline (Sequential Workflow)

The main pipeline follows a naming convention where scripts are prefixed with letters/numbers indicating execution order:

1. **Build initial dataset from raw snapshots** (a00):
   ```bash
   python a00build_dataset_with_features.py
   ```
   - Uses `active_process_dataset` from config.yaml to select which dataset to process
   - Loads raw CSP snapshots from data_dir (e.g., `option/put/put25_1027-1107/coveredPut_*.csv`)
   - Merges multiple snapshot files into single dataset
   - Merges GEX features from `gex101/` (symlink to NAS)
   - Adds VIX and macro price features
   - **Automatically extracts and writes unique symbols** to `symbols_in_option_data_*.txt`
   - Output: Merged trades file (data_basic_csv) + enriched dataset + symbols file
   - **Symbol extraction is now automated** - no manual step needed!
   - Note: File will be renamed to a03 in future to match execution order

   **Batch processing all datasets**:
   ```bash
   python a00build_all_datasets.py
   ```
   - Processes ALL datasets in common_configs (instead of just active_process_dataset)
   - Useful for bulk regeneration

2. **Collect corporate events** (a01):
   ```bash
   python a01_collect_corp_events.py
   ```
   - Reads symbol list from `symbols_in_option_data_*.txt`
   - Scrapes EDGAR for earnings (8-K Item 2.02 filings) for those symbols
   - Fetches stock splits from yfinance API
   - Unified output with both event types
   - Output: `output/data_prep/corp_events/events_*.csv`
   - **Run once per dataset** (events don't change)

3. **Filter trades near corporate events** (a02):
   ```bash
   python a02_filter_noisy_trades.py
   ```
   - Reads events from a01 output and trades from data_basic_csv
   - Removes trades opened or expiring too close to earnings/splits
   - Configurable exclusion windows (±days before/after events)
   - Reduces training noise from abnormal volatility
   - Output: `option/put/filtered/trades_filtered_*.csv`
   - See `doc/a02_filter_usage.md` for tuning guide

4. **Label trades** (a09):
   ```bash
   python a09label_data.py
   ```
   - Fetches expiry-date closing prices via yfinance
   - Computes PnL, return_pct, return_mon (per-month), return_ann (annualized)
   - Labels: `win` = (return_pct > 0)
   - Uses `cutoff_date` from config to avoid labeling unexpired trades
   - Output: `output/data_labeled/labeled_trades_*.csv`

5. **Train Winner Classifier** (b01):
   ```bash
   python b01train_winner_classifier_pct_oof.py
   ```
   - Out-of-fold (OOF) cross-validation with TimeSeriesSplit or StratifiedKFold
   - Model types: lgbm (default), catboost, rf (RandomForest)
   - Sample weighting by return magnitude
   - Outputs: model pkl, feature importances, threshold tables, PR curves
   - Output dir: `output/winner_train/v9d_oof_*/`

6. **Score new candidates**:
   ```bash
   python score_winner_classifier_env.py
   ```
   - Loads trained model from `WINNERSCORE_MODEL_IN`
   - Scores unlabeled or new trade data
   - Applies thresholds for win_predict column
   - Output: `output/winner_score/*/scores_winner_*.csv`

7. **Train Tail Loss Classifier** (optional):
   ```bash
   python train_tail_with_gex.py
   ```
   - Labels tail losses as worst K% by dollar PnL (default 5%)
   - Uses GradientBoosting with stratified CV
   - Output: `output/tails_train/*/tail_model_*.pkl`

8. **Evaluate models**:
   ```bash
   python eval_binary_classifier_env.py
   ```
   - Computes ROC-AUC, PR-AUC, confusion matrices
   - Supports both winner and tail classifiers
   - Generates threshold sweep tables

### Testing
```bash
pytest test/
```

## Configuration System

### Unified Configuration (Updated 2026-01-30)

The project uses a **unified template-based configuration system**. Change 3 variables at the top of `config.yaml` to control the entire pipeline.

**Active Configuration (Top of config.yaml)**:
```yaml
active_train_profile: "origabcde"   # Which datasets to train on
active_score_dataset: "f"           # Which dataset to score
active_process_dataset: "f"         # Which dataset to process (a00/a01/a02)
```

**Important**: `active_process_dataset` controls the entire data preparation pipeline:
- **a00**: Processes the specified dataset, extracts symbols automatically
- **a01**: Uses the symbols file from a00 to collect corporate events
- **a02**: Filters trades using events from a01

**Key Features**:
- **Template resolution**: Paths use `{active_train_profile}`, `{active_score_dataset}`, `{active_process_dataset}` placeholders
- **Automatic sync**: Change one variable, all paths update automatically
- **Walk-forward validation**: Profile names show incremental growth (orig → origa → origab → origabc)
- **Dataset registry**: All dataset configs in one place with naming convention enforcement

### config.yaml Structure

The project uses YAML anchors (`&anchor` / `*anchor`) plus template-based paths:

- **Active variables** (top of file): Control which datasets to use
  - `active_train_profile`: Training dataset combination (e.g., "origabcde")
  - `active_score_dataset`: Scoring dataset (e.g., "f")
  - `active_process_dataset`: Processing dataset for a00/a01/a02 (e.g., "f")

- **common_configs**: Dataset registry with all dataset-specific settings
  - `data_dir`: Where raw CSP snapshots live
  - `data_basic_csv`: Merged raw trades file
  - `output_csv`: Labeled trades destination
  - `cutoff_date`: Don't label trades expiring after this (prevents peeking at future)
  - `events_start_date`, `events_end_date`: Corp events date range (for a01)
  - `events_output`: Corp events output CSV path (for a01)
  - `tickers_file`: Symbol list for corp events collection (for a01)
  - `filtered_trades_csv`, `filtered_out_csv`: Filtered trade paths (for a02)

- **winner**: Winner classifier settings with templates
  - `input`: `"output/data_labeled/labeled_merged_with_gex_macro_{active_train_profile}.csv"`
  - `output_dir`: `"output/winner_train/v9_oof_{active_train_profile}"`
  - `train_target`: "return_mon" (monthly), "return_ann" (annualized), or "return_pct"
  - `model_type`: "lgbm", "catboost", or "rf"
  - `oof_folds`: Number of cross-validation folds
  - `time_series`: Use TimeSeriesSplit (1) or StratifiedKFold (0)

- **winnerscore**: Scoring configuration with templates
  - `score_input`: `"output/data_labeled/labeled_trades_with_gex_macro_{active_score_dataset}.csv"`
  - `model_in`: `"output/winner_train/v8_oof_{active_train_profile}/winner_classifier_model_{active_train_profile}_lgbm.pkl"`
  - `score_out_folder`: `"output/winner_score/v8_model_{active_train_profile}"`
  - `score_out`: `"scores_winner_lgbm_{active_score_dataset}.csv"`

### corp_action_config.yaml

Common behavior settings for a01/a02 (dataset-specific paths come from config.yaml):
- **SEC settings**: `user_agent`, `sleep_seconds`, `cache_dir`
- **Exclusion windows**: Days before/after earnings/splits to filter trades
- **Column names**: CSV column mappings

### Environment Variables

`.env` files override config.yaml settings. Common patterns:
- Training uses `WINNER_*` prefixed variables
- Scoring uses `WINNERSCORE_*` prefixed variables
- Tail model uses `TAIL_*` and `CSV_INPUT`, `MODEL_OUT`

The `service/env_config.py` module:
- Provides `getenv(key, default)` that checks YAML first, then falls back to environment variables
- Resolves `{template}` placeholders in paths automatically
- Provides `get_active_dataset_config()` to fetch dataset-specific settings for a00/a01/a02

**Note**: The `common:` section with `<<: *anchor` is legacy. Modern scripts (a00/a01/a02) now use `active_process_dataset` and `get_active_dataset_config()` directly, making the common section less relevant.

**See**: `doc/unified_config_usage.md` for detailed usage guide and examples.

## Architecture

### Core Service Modules (`service/`)

- **env_config.py**: Unified config loader (YAML + .env fallback)
- **data_prepare.py**: Price caching, capital calculations, macro feature engineering
- **preprocess.py**: DTE calculation, normalized returns, GEX merging
- **winner_scoring.py**: Threshold selection, calibration, prediction logic
- **production_data.py**: Feature engineering for live/scoring data
- **nasdaq_earnings.py**: Earnings calendar scraping
- **stock_data_manager2.py**: Batch price updates with caching

### Data Flow

```
Raw CSP snapshots (option/put/put25_*/coveredPut_*.csv)
  ↓ [a00build_dataset_with_features.py]
Merged trades + enriched dataset (data_basic_csv + trades_with_gex_macro_*.csv)
  ↓ [Extract unique symbols to symbols_in_option_data_*.txt]
Symbol list (output/data_prep/corp_events/symbols_in_option_data_*.txt)
  ↓ [a01_collect_corp_events.py]
Corporate events (output/data_prep/corp_events/events_*.csv)
  ↓ [a02_filter_noisy_trades.py reads data_basic_csv + events]
Filtered trades (option/put/filtered/trades_filtered_*.csv)
  ↓ [a09label_data.py + yfinance price lookups]
Labeled dataset (output/data_labeled/labeled_trades_*.csv)
  ↓ [b01train_winner_classifier_pct_oof.py]
Trained model (output/winner_train/*/winner_classifier_*.pkl)
  ↓ [score_winner_classifier_env.py]
Scored candidates (output/winner_score/*/scores_winner_*.csv)
```

### Feature Engineering

Three feature groups (from `service/utils.py`):
- **BASE_FEATS**: Core option greeks (delta, moneyness, IVR, BEP, potential return, volume, OI)
- **GEX_FEATS**: Gamma exposure metrics merged from `gex101/` directory by date + symbol
- **NEW_FEATS**: VIX, underlying price momentum (2d, 5d returns), normalized returns

### Model Types

1. **Winner Classifier**: Binary classification for profitability
   - Label: `return_pct > 0` (or `return_mon > epsilon` with configurable threshold)
   - Models: LightGBM (primary), CatBoost, RandomForest
   - Training: OOF cross-validation with sample weighting by |return|
   - Threshold tuning: Target precision (e.g., 0.88, 0.92) or best F1

2. **Tail Loss Classifier**: Predict worst K% of trades by PnL
   - Label: Bottom K% quantile (default 5%) by dollar loss
   - Model: GradientBoosting with stratified CV
   - Use case: Filter out catastrophic losses before applying winner model

### Price Data Management

- **Caching**: Parquet files in `output/price_cache/`
- **Batch updates**: `daily_stock_update.py` refreshes price cache
- **Missing stocks**: `data/missing_stocks.json` tracks symbols to skip
- **Cutoff dates**: Configured per dataset to prevent future leakage

### Cutoff Date Logic

Critical anti-leakage measure:
- Each dataset config has a `cutoff_date`
- Labeling script (`a09label_data.py`) only processes trades with `expirationDate + 1 day <= cutoff_date`
- Scoring on recent data uses trades that haven't expired yet
- Price cache respects cutoff to avoid using future prices

### Corporate Events Filtering (a02)

New noise-reduction step:
- **Purpose**: Remove trades near earnings/splits that exhibit abnormal behavior
- **Configuration**: `corp_action_config.yaml` exclusion_windows section
- **Two-phase filtering**:
  1. Trade phase: Exclude if tradeTime too close to event
  2. Expiry phase: Exclude if expirationDate too close to event
- **Event types**:
  - EARNINGS: From EDGAR 8-K Item 2.02 filings (semantic filtering to avoid false positives)
  - SPLIT: From yfinance splits API
- **Typical impact**: 10-20% of trades excluded
- **Tuning**: Adjust `days_before_*` and `days_after_*` parameters based on model performance
- See `doc/a02_filter_usage.md` for detailed guide

## Common Development Patterns

### Walk-Forward Validation with a07

The `a07merge_dataset_with_features.py` script supports incremental training:

```bash
python a07merge_dataset_with_features.py
```

**Purpose**: Rolling window strategy where test data becomes training data as time progresses.

**How it works:**
1. Config defines dataset groups with tags: `orig`, `a`, `b`, `c`, etc.
2. Script creates incremental merges: `orig`, `orig+a`, `orig+a+b`, etc.
3. Each non-merged dataset serves as out-of-sample validation
4. After validation, merge it into training set and use next period for testing

**Example workflow:**
- Train on `orig` (Apr-Aug), validate on `a` (Aug-Sep)
- If good, retrain on `orig+a`, validate on `b` (Sep-Oct)
- Continue rolling forward...

This prevents look-ahead bias and simulates real-time model retraining.

### Adding a New Dataset

1. Add entry to `common_configs` in config.yaml:
   ```yaml
   oct_27: &oct_27_config
     data_dir: "option/put/put25_1027-1107"
     data_basic_csv: "trades_raw_f_1027.csv"
     output_csv: "labeled_trades_f_1027.csv"
     cutoff_date: "2025-11-15"
   ```

2. Update `common` section to use new config:
   ```yaml
   common:
     <<: *oct_27_config
   ```

3. Run pipeline: build → label → train

### Switching Model Types

Edit config.yaml:
```yaml
winner:
  model_type: "catboost"  # or "lgbm" or "rf"
```

Model-specific parameters are handled in the training script.

### Running Incremental Training

To train on combined datasets:
1. Label each dataset separately with appropriate cutoff_date
2. Concatenate labeled CSVs manually or use merge scripts
3. Point `winner.input` to merged CSV
4. Train with `time_series: 1` to respect temporal ordering

## File Naming Conventions

- `a##*.py`: Data preparation/feature engineering (sequential pipeline steps)
  - `a01_collect_corp_events.py`: Corporate events collection
  - `a02_filter_noisy_trades.py`: Event proximity filtering
  - `a00build_dataset_with_features.py`: (will become a03) Feature engineering
  - `a09label_data.py`: Outcome labeling
- `b##*.py`: Model training scripts
- `*_env.py`: Reads from .env files
- `*_oof.py`: Out-of-fold cross-validation variant
- Files without prefixes: Standalone utilities or evaluation scripts

## Service Modules

- `service/split_detector.py`: Stock split detection via yfinance API
- `service/nasdaq_earnings.py`: Nasdaq earnings calendar scraper (alternative/supplementary to EDGAR)
- `service/data_prepare.py`: Price caching, capital calculations, macro features
- `service/preprocess.py`: DTE calculation, normalized returns, GEX merging
- `service/winner_scoring.py`: Threshold selection and prediction logic
- `service/production_data.py`: Feature engineering for live scoring

## Output Directory Structure

```
output/
├── data_prep/           # Enriched datasets (pre-labeling)
├── data_labeled/        # Labeled datasets with returns
├── winner_train/        # Trained winner models
├── winner_score/        # Scored candidates
├── tails_train/         # Tail loss models
├── tails_score/         # Tail loss predictions
├── eval/                # Evaluation reports
├── price_cache/         # Cached yfinance data (parquet)
└── vix_data.csv         # VIX historical data
```

## Important Notes

- **GEX data**: Symlinked from NAS at `gex101 -> /mnt/nas_share/dev/data/gex101/processed/csv`
- **Time zones**: All dates normalized to midnight; exchange_calendars used for session resolution
- **Missing values**: Median imputation on train split only (avoid leakage)
- **Sample weighting**: Optional (WINNER_USE_WEIGHTS=1) to emphasize larger magnitude returns
- **Threshold calibration**: Can auto-calibrate on validation set or use fixed thresholds from training

## Dependencies

Key libraries:
- pandas, numpy, scipy
- scikit-learn (RandomForest, GradientBoosting)
- lightgbm (primary winner model)
- catboost (alternative model)
- yfinance (price data)
- pyarrow, fastparquet (efficient caching)
- exchange_calendars (trading day logic)
- python-dotenv (config)

## Testing

Test suite in `test/` covers:
- Data manipulation and filtering
- Feature engineering correctness
- Model scoring refactoring verification
- Price cache management
- Corporate events handling

Run specific test:
```bash
pytest test/test_preprocess.py -v
```
