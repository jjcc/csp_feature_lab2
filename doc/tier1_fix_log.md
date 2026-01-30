# Tier 1 Fixes - Implementation Log

**Date**: 2026-01-30
**Status**: ✅ Complete
**Time**: Approximately 1 hour

---

## Overview

All 4 critical fixes from the refactor plan have been successfully implemented. These fixes address data integrity, reproducibility, and maintainability issues identified in the comprehensive code review.

---

## 1. Fixed `env_config.py` Fallback Logic ✅

### Issue
- Environment variable fallback was commented out (lines 43-44)
- The `default` parameter was completely ignored
- Always returned `None` for missing keys instead of checking env vars or using defaults

### Files Modified
- `service/env_config.py`

### Changes Made
```python
# BEFORE (lines 37-48)
def get(self, key, default=None):
    if self._config is None:
        yaml_config = self._load_yaml_config()
        self._config = self._flatten_config(yaml_config)

    #if self._env is None:
    #    self._env = self._load_env_config()

    # Try YAML config first, then env variables, then default
    #return self._config.get(key, self._env.get(key, default))
    return self._config.get(key, None)  # BROKEN - ignores default!

# AFTER (lines 37-50)
def get(self, key, default=None):
    if self._config is None:
        yaml_config = self._load_yaml_config()
        self._config = self._flatten_config(yaml_config)

    # Try YAML config first
    value = self._config.get(key)

    # If not found and fallback enabled, try environment variables
    if value is None and self.fallback_to_env:
        value = os.getenv(key)

    # Return value if found, otherwise return default
    return value if value is not None else default
```

### Impact
- ✅ Configuration system now works as designed
- ✅ Proper fallback chain: YAML → env vars → default value
- ✅ Easier debugging of missing configuration values

---

## 2. Consolidated Winner Scoring Modules ✅

### Issue
- Two implementations of the same functionality:
  - `service/winner_scoring.py` (6.2 KB, basic version)
  - `service/winner_scoring_refactor.py` (12 KB, improved version)
- Different scripts imported from different modules
- Refactored version had better error handling but was never fully adopted

### Files Modified
- `service/winner_scoring.py` - Completely replaced with better version
- `service/winner_scoring_refactor.py` - Deleted

### Changes Made
**Replaced `winner_scoring.py` with improved version that includes:**
- Better error handling with specific exceptions
- Input validation (empty DataFrames, invalid thresholds, missing columns)
- Logging support via Python's logging module
- Type hints with Protocol for ClassifierModel
- Module-level constants for default values
- Comprehensive docstrings with Args/Returns/Raises sections
- Backward compatibility aliases for deprecated functions

**Key improvements:**
```python
# NEW: Input validation
def score_winner_data(df: pd.DataFrame, model_pack: WinnerModelPack,
                     proba_col: str = DEFAULT_PROBA_COL):
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    if not proba_col or not isinstance(proba_col, str):
        raise ValueError("proba_col must be a non-empty string")

    # ... existing logic with try-except ...
    try:
        proba = model_pack.model.predict_proba(X)[:, 1]
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        raise

# NEW: Threshold validation
def apply_winner_threshold(df: pd.DataFrame, proba_col: str, pred_col: str,
                          threshold: float) -> pd.DataFrame:
    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    if proba_col not in df.columns:
        raise ValueError(f"Column '{proba_col}' not found in DataFrame")
```

### Impact
- ✅ Single source of truth for winner scoring logic
- ✅ Better error messages make debugging easier
- ✅ Type safety via Protocol for sklearn-like models
- ✅ All imports continue to work (backward compatible)

---

## 3. Replaced Hard-Coded Paths with pathlib ✅

### Issue
- Hard-coded relative paths break when running from different directories
- Examples:
  - `"data/missing_stocks.json"`
  - `"data/wolf/WOLF.parquet"`
  - `"option/put/coveredPut_*.csv"`
  - `"log/processed.log"`
  - `"output/tails_train/v6b_ne/tail_model_gex_v6b_ne_cut05.pkl"`

### Files Modified
1. `service/data_prepare.py`
2. `task_score_tail_winner.py`
3. `a09label_data.py`

### Changes Made

**`service/data_prepare.py`:**
```python
# Added at top of file
from pathlib import Path

# Project root for absolute paths
PROJECT_ROOT = Path(__file__).parent.parent
MISSING_STOCKS_PATH = PROJECT_ROOT / "data" / "missing_stocks.json"
WOLF_PRICE_PATH = PROJECT_ROOT / "data" / "wolf" / "WOLF.parquet"

# Updated usage
with open(MISSING_STOCKS_PATH, "r") as f:  # Was: "data/missing_stocks.json"
    missing_stocks = json.load(f)

df = pd.read_parquet(WOLF_PRICE_PATH)  # Was: "data/wolf/WOLF.parquet"
```

**`task_score_tail_winner.py`:**
```python
# Added at top
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
TAIL_MODEL_IN = PROJECT_ROOT / "output" / "tails_train" / "v6b_ne" / "tail_model_gex_v6b_ne_cut05.pkl"
WINNER_MODEL_IN = PROJECT_ROOT / "output" / "winner_train" / "v7_oof_ne_ts_w_lgbm_tr_ts" / "winner_classifier_v7_lgbm.pkl"
OPTION_DATA_DIR = PROJECT_ROOT / "option" / "put"
PROCESS_LOG_PATH = PROJECT_ROOT / "log" / "processed.log"

# Updated usage
if not PROCESS_LOG_PATH.exists():
    PROCESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESS_LOG_PATH.touch()

latest_file_with_path = max(glob(str(OPTION_DATA_DIR / glob_pat)), key=os.path.getctime)

out_dir = PROJECT_ROOT / "prod" / "output"
out_dir.mkdir(parents=True, exist_ok=True)
```

**`a09label_data.py`:**
```python
# Added at top
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MISSING_STOCKS_PATH = PROJECT_ROOT / "data" / "missing_stocks.json"
EXCLUDE_STOCKS_PATH = PROJECT_ROOT / "data" / "exclude_stocks.json"

# Updated usage
with open(MISSING_STOCKS_PATH, "r") as fp:
    missing_stocks = json.load(fp)
with open(EXCLUDE_STOCKS_PATH, "r") as fp:
    exclude_stocks = json.load(fp)
```

### Impact
- ✅ Scripts work from any working directory
- ✅ More portable across different environments
- ✅ Clearer path construction with `/` operator
- ✅ Automatic parent directory creation with `mkdir(parents=True)`

---

## 4. Added Pipeline Input Validation ✅

### Issue
- Pipeline scripts silently filled missing columns with NaN
- Errors discovered late in the pipeline (during training or scoring)
- No clear indication of what went wrong when columns were missing
- Difficult to debug data quality issues

### Files Modified
1. `a09label_data.py`
2. `b01train_winner_classifier_pct_oof.py`
3. `score_winner_classifier_env.py`

### Changes Made

**`a09label_data.py` - Labeling validation:**
```python
def build_dataset(raw: pd.DataFrame, max_rows: int = 0, preload_closes: dict = None) -> pd.DataFrame:
    """
    Prepare labeled dataset for modeling.
    Missing optional columns are filled with NaN.
    """
    df = raw.copy()

    # NEW: Validate required columns exist
    required_cols = ["baseSymbol", "expirationDate", "strike", "tradeTime", "underlyingLastPrice"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}. Cannot proceed with labeling.")

    # Standardize expected columns (optional ones can be filled with NaN)
    optional_cols = [
        "delta", "moneyness", "impliedVolatilityRank1y", "potentialReturn",
        "potentialReturnAnnual", "breakEvenProbability", "percentToBreakEvenBid",
        "openInterest", "volume", "__source_file"
    ]
    for c in optional_cols:
        if c not in df.columns:
            df[c] = np.nan
```

**`b01train_winner_classifier_pct_oof.py` - Training validation:**
```python
# Load and prepare data
input_csv = config.input_csv
df = pd.read_csv(config.input_csv)

# NEW: Validate required columns exist
required_cols = ["captureTime", "symbol", config.train_target]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in input CSV: {missing_cols}. "
                    f"Available columns: {list(df.columns)}")
```

**`score_winner_classifier_env.py` - Scoring validation:**
```python
def load_and_preprocess_data(config: ScoringConfig) -> pd.DataFrame:
    """Load and preprocess input data."""
    df = pd.read_csv(config.csv_in)

    # NEW: Validate required columns exist
    required_cols = ["symbol", "tradeTime"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSV: {missing_cols}. "
                        f"Available columns: {list(df.columns)}")
```

### Impact
- ✅ Fail-fast with clear, actionable error messages
- ✅ Easier to identify data preparation issues
- ✅ Shows both missing and available columns for debugging
- ✅ Prevents wasted compute time on invalid data

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Files modified | 8 |
| Files deleted | 1 |
| Lines of code changed | ~150 |
| Critical bugs fixed | 4 |
| Hard-coded paths removed | 10+ |
| Validation checks added | 3 |

---

## Before vs After

### Before
❌ Config fallback broken → hard to debug missing values
❌ 2 winner_scoring modules → confusion about which to use
❌ Hard-coded paths → breaks from different directories
❌ Silent failures → data issues discovered late in pipeline

### After
✅ Config fallback working → proper default handling
✅ 1 winner_scoring module → clear, well-documented API
✅ Path-based paths → portable, works from any directory
✅ Explicit validation → immediate, actionable error messages

---

## Files Modified (Complete List)

1. **service/env_config.py** - Fixed fallback logic in `get()` method
2. **service/winner_scoring.py** - Replaced with improved version (better error handling, validation, logging)
3. **service/winner_scoring_refactor.py** - Deleted (consolidated into winner_scoring.py)
4. **service/data_prepare.py** - Added PROJECT_ROOT, MISSING_STOCKS_PATH, WOLF_PRICE_PATH constants
5. **task_score_tail_winner.py** - Converted all hard-coded paths to pathlib Path objects
6. **a09label_data.py** - Added PROJECT_ROOT, path constants, and required column validation
7. **b01train_winner_classifier_pct_oof.py** - Added input validation for training data
8. **score_winner_classifier_env.py** - Added input validation for scoring data

---

## Testing Notes

### Manual Verification
- ✅ All modified files have valid Python syntax
- ✅ Imports resolve correctly
- ✅ Path construction uses PROJECT_ROOT pattern consistently
- ✅ Validation error messages are clear and actionable

### Known Limitations
- Unit tests for env_config.py don't exist yet (should be added in Tier 3)
- Full pipeline test needed to verify end-to-end behavior
- Some test files still have hard-coded paths (not critical for production)

---

## Next Steps (Tier 2 Priorities)

Based on the refactor plan, the next recommended fixes are:

### High Priority
1. **Consolidate duplicate utility functions**
   - Files: `service/utils.py` and `service/model_service.py`
   - Functions duplicated: `prep_tail_training_df()`, `fill_features_with_training_medians()`, `prep_winner_like_training()`
   - Estimated effort: 1-2 hours

2. **Clean up dead code**
   - Delete: `service/stock_data_manager.py` (replaced by v2)
   - Delete: `service/identity_cal.py` (empty module)
   - Delete: `b01train_winner_classifier_pct_oof_fix.py` (if not canonical)
   - Rename: `service/stock_data_manager2.py` → `service/stock_data_manager.py`
   - Estimated effort: 30 minutes

3. **Rename pipeline scripts for clarity**
   - `edga_events_scrap.py` → `a01_collect_corp_events.py`
   - Clarify `a00build_dataset_with_features.py` vs `a03merge_fundamentals_events.py`
   - Remove `_fix` suffix from training scripts
   - Estimated effort: 1 hour

4. **Break up monolithic scripts**
   - `b01train_winner_classifier_pct_oof.py` (731 lines) → extract functions
   - `edga_events_scrap.py` (500+ lines) → split EDGAR/splits logic
   - Estimated effort: 3-4 hours

---

## Lessons Learned

1. **Pathlib is superior to string concatenation** - More readable, cross-platform, safer
2. **Validation should happen early** - Fail-fast prevents wasted computation
3. **Consolidation reduces cognitive load** - One well-documented module beats two unclear ones
4. **Configuration fallback chains are subtle** - Easy to break, hard to debug without tests

---

## Success Criteria Met ✅

- [x] All Tier 1 fixes from refactor plan implemented
- [x] No breaking changes to existing imports
- [x] Backward compatibility maintained where needed
- [x] Error messages are clear and actionable
- [x] Code is more maintainable and portable
- [x] Documentation updated (refactor_plan.md, this log)

---

**End of Tier 1 Implementation**
**Ready for Tier 2 when you are!** 🚀
