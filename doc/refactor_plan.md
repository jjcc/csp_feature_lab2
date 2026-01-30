# Code Structure Review - CSP ML Pipeline

Generated: 2026-01-30

## 🔴 **Critical Issues** (Fix These First)

### 1. **Broken Configuration Fallback**
`service/env_config.py:48` - The environment variable fallback is broken:
```python
def get(self, key, default=None):
    return self._config.get(key, None)  # Ignores 'default' parameter and env vars!
```
This should return `default` when key is missing and check environment variables.

### 2. **Duplicate Winner Scoring Modules**
You have **two implementations** of the same functionality:
- `service/winner_scoring.py` (6.2 KB)
- `service/winner_scoring_refactor.py` (12 KB)

Both define identical functions. The refactored version has better error handling but they're never consolidated.

### 3. **Duplicate Utility Functions**
`service/utils.py` and `service/model_service.py` contain **identical** functions:
- `prep_tail_training_df()`
- `fill_features_with_training_medians()`
- `prep_winner_like_training()`

Different scripts import from different modules, creating maintenance nightmares.

### 4. **Hard-Coded Paths** (breaks portability)
```python
# service/data_prepare.py:49
with open("data/missing_stocks.json", "r")  # Relative path, breaks if CWD changes

# task_score_tail_winner.py
TAIL_MODEL_IN = "output/tails_train/v6b_ne/tail_model_gex_v6b_ne_cut05.pkl"
```

---

## 🟡 **Major Structural Issues**

### 5. **Pipeline Naming Confusion**
- `edga_events_scrap.py` should be `a01_collect_corp_events.py` (documented as a01 but never renamed)
- Both `a00build_dataset_with_features.py` and `a03merge_fundamentals_events.py` exist - unclear which is canonical
- `b01train_winner_classifier_pct_oof_fix.py` vs original - which is current?

### 6. **Dead Code**
- `stock_data_manager.py` (5.3 KB) - never imported, replaced by `stock_data_manager2.py`
- `identity_cal.py` (140 bytes) - empty module
- Duplicate training scripts (`b01...` and `b01..._fix.py`)

### 7. **Large Monolithic Scripts**
- `b01train_winner_classifier_pct_oof.py`: **731 lines** - should be broken into:
  - Config loading
  - Data preparation
  - Training/CV loop
  - Model artifact saving

- `edga_events_scrap.py`: **500+ lines** mixing EDGAR fetching, ticker management, and splits

### 8. **Unsafe Exception Handling**
```python
# service/env_config.py
except:  # Bare except - catches ALL exceptions including KeyboardInterrupt
    pass

# service/data_prepare.py
except Exception:
    print(f"[WARN] cache write failed for {symbol}: {e}")  # 'e' is undefined here!
```

---

## 📊 **Detailed Analysis**

### Configuration Management Issues

**Current Implementation:**

`config.yaml` structure:
```yaml
common_configs:
  original: &original_config
  aug_11: &aug_11_config
  sep_1: &sep_1_config
  ...7 dataset variants...
common:
  <<: *original_config  # Single active config
```

**Problems:**
1. **Broken Fallback Logic** - env variables never checked despite `fallback_to_env=True`
2. **Complex YAML Structure** - 7+ dataset configs but only one active at a time
3. **Inconsistent Usage** - some scripts use `.yaml`, others use `.env`
4. **Magic Environment Variable Names** - no validation or defaults
5. **Dataset-Specific Overrides** - `labeling` section uses different config than `common`

### Service Module Duplication

| Module | Size | Status | Issue |
|--------|------|--------|-------|
| `winner_scoring.py` | 6.2 KB | In use | Duplicate of refactor version |
| `winner_scoring_refactor.py` | 12 KB | Better version | Not fully adopted |
| `stock_data_manager.py` | 5.3 KB | Dead code | Replaced by v2 |
| `stock_data_manager2.py` | 10.3 KB | In use | Should rename to remove "2" |
| `utils.py` | 10.8 KB | In use | Mixed responsibilities |
| `model_service.py` | 6.5 KB | In use | Duplicates utils.py |
| `adapters_winner.py` | 0.8 KB | Thin wrapper | Could be inlined |
| `adapters_tail.py` | 0.8 KB | Thin wrapper | Could be inlined |
| `identity_cal.py` | 140 B | Empty | Should delete |

### Data Pipeline Flow Issues

**Documented Pipeline:**
```
Raw CSP → (a01 edga_events) → Corp Events
         → (a02 filter_noisy) → Filtered Trades
         → (a00→a03 build) → Enriched Dataset
         → (a09 label) → Labeled Dataset
         → (b01 train) → Model
         → (score) → Predictions
```

**Actual Problems:**
1. **a01 Naming Mismatch** - Script is `edga_events_scrap.py` (no a## prefix)
2. **a03 Confusion** - Both `a00build_dataset_with_features.py` and `a03merge_fundamentals_events.py` exist
3. **Path Construction** - Fragile string concatenation with `/`
4. **Missing Validation** - No schema checks at pipeline boundaries
5. **Cutoff Date Logic** - Duplicated across multiple files

### Code Quality Issues

**Hard-Coded Paths** (7+ occurrences):
```python
"data/missing_stocks.json"
"data/wolf/WOLF.parquet"
"output/tails_train/v6b_ne/tail_model_gex_v6b_ne_cut05.pkl"
```

**Magic Numbers** (10+ instances):
```python
if start_dt.weekday() == 5:  # What is 5?
    start_dt += pd.Timedelta(days=2)
sleep(1)  # Why 1 second?
min_ratio_change: float = 0.1  # Why 0.1?
```

**Large Monolithic Functions:**
- `b01train_winner_classifier_pct_oof.py`: 731 lines
- `edga_events_scrap.py`: 500+ lines
- `a09label_data.py`: 400+ lines

**Broad Exception Handling:**
```python
except:  # Bare except - dangerous!
    pass

except Exception:  # Too broad
    print(f"Error: {e}")  # e undefined
```

---

## 📈 **Recommendations by Priority**

### **TIER 1: Critical Fixes (Do Now)**

1. **Fix `env_config.py` fallback logic**
   ```python
   def get(self, key, default=None):
       val = self._config.get(key)
       if val is None and self._fallback_to_env:
           val = os.getenv(key)
       return val if val is not None else default
   ```

2. **Consolidate winner_scoring modules**
   - Delete `winner_scoring.py`
   - Rename `winner_scoring_refactor.py` → `winner_scoring.py`
   - Update all imports

3. **Replace hard-coded paths with pathlib**
   ```python
   from pathlib import Path

   PROJECT_ROOT = Path(__file__).parent.parent
   MISSING_STOCKS = PROJECT_ROOT / "data" / "missing_stocks.json"
   ```

4. **Add pipeline input validation**
   - Check required columns exist before processing
   - Raise errors instead of silent NaN fills

---

### **TIER 2: Structural Improvements**

5. **Consolidate duplicate functions**
   - Pick ONE location for shared utilities (probably `service/utils.py`)
   - Delete duplicates from `model_service.py`
   - Create `service/constants.py` for BASE_FEATS, GEX_FEATS, etc.

6. **Clean up dead code**
   ```bash
   # Remove these files:
   - service/stock_data_manager.py (replaced by stock_data_manager2.py)
   - service/identity_cal.py (empty)
   - b01train_winner_classifier_pct_oof_fix.py (if it's not canonical)
   ```

7. **Rename pipeline scripts for clarity**
   ```bash
   mv edga_events_scrap.py a01_collect_corp_events.py
   # Clarify a00 vs a03 - merge or make purpose explicit
   ```

8. **Break up monolithic scripts**
   - Extract functions from 500+ line scripts
   - Create service modules for reusable logic
   - Keep main scripts as thin orchestrators

---

### **TIER 3: Quality Improvements**

9. **Replace magic numbers**
   ```python
   WEEKDAY_SATURDAY = 5
   WEEKDAY_SUNDAY = 6
   DEFAULT_SLEEP_SECONDS = 1.0
   MIN_RATIO_CHANGE = 0.1
   ```

10. **Improve exception handling**
    - Use specific exception types
    - Add proper logging (not print statements)
    - Create domain-specific exceptions

11. **Add integration tests**
    - Test full pipeline flow
    - Test config loading with missing files
    - Test data validation at boundaries

12. **Document architecture**
    - Create `ARCHITECTURE.md` mapping module responsibilities
    - Document data flow: stage → input → output → required columns

---

## 🏗️ **Proposed New Structure**

```
csp_feature_lab2/
├── config/
│   ├── config.yaml
│   ├── corp_action_config.yaml
│   └── constants.py          # NEW: Feature lists, magic numbers
├── pipeline/                  # NEW: Rename root scripts
│   ├── a01_collect_corp_events.py
│   ├── a02_filter_noisy_trades.py
│   ├── a03_build_features.py
│   ├── a04_label_outcomes.py
│   └── b01_train_winner.py
├── service/
│   ├── core/                 # NEW: Core utilities
│   │   ├── env_config.py
│   │   ├── paths.py         # Centralized path management
│   │   └── utils.py         # Keep only generic utils
│   ├── data/
│   │   ├── price_cache.py   # Consolidate price loading
│   │   ├── corp_events.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── winner_scoring.py  # Single version
│   │   └── tail_scoring.py
│   └── features/
│       └── engineering.py
├── scripts/                  # Ad-hoc scripts not in main pipeline
│   ├── daily_stock_update.py
│   └── scan_price_cache_dates.py
├── test/
└── archive/                  # OLD: Move deprecated files here
```

---

## 🎯 **Recommended Action Plan**

### **Week 1: Critical Fixes**
- [ ] Fix `env_config.py` fallback logic
- [ ] Consolidate winner_scoring modules
- [ ] Replace hard-coded paths in top 5 files
- [ ] Add input validation to a09label_data.py

### **Week 2: Deduplication**
- [ ] Consolidate duplicate utility functions
- [ ] Remove dead code (stock_data_manager, identity_cal)
- [ ] Clarify/rename training script versions
- [ ] Create service/constants.py for feature lists

### **Week 3: Refactoring**
- [ ] Break up 500+ line scripts into functions
- [ ] Rename pipeline scripts (a01, a03 clarity)
- [ ] Move ad-hoc scripts to separate directory
- [ ] Consolidate price loading functions

### **Week 4: Quality**
- [ ] Add input validation to all pipeline stages
- [ ] Replace magic numbers with constants
- [ ] Improve exception handling (specific types)
- [ ] Write integration tests

---

## 📊 **Summary Statistics**

| Metric | Value | Assessment |
|--------|-------|------------|
| Root-level Python files | 64 | Too many (should be <20) |
| Service modules | 16 | Good (modular) but too many overlaps |
| Duplicate code sections | 4+ major | Refactoring debt |
| Test files | 17 | Reasonable coverage but weak integration tests |
| Lines in largest script | 731 | Refactoring needed |
| Config variants | 7 | Too many for single-choice system |
| Hard-coded paths | 7+ | All should be config-driven |
| Broken fallback logic | 1 major | `env_config.py` line 48 |

---

## 🔍 **Specific File Recommendations**

### High Priority Files to Refactor

1. **service/env_config.py** (4.9 KB)
   - Fix fallback logic immediately
   - Add validation for required keys
   - Document expected config structure

2. **service/winner_scoring.py + winner_scoring_refactor.py** (18.2 KB total)
   - Keep refactored version only
   - Rename to remove "refactor" suffix
   - Update all imports

3. **service/utils.py** (10.8 KB)
   - Extract feature lists to constants.py
   - Remove duplicate functions
   - Split into domain-specific modules

4. **b01train_winner_classifier_pct_oof.py** (731 lines)
   - Extract config loading → function
   - Extract data prep → function
   - Extract CV loop → function
   - Extract artifact saving → function

5. **edga_events_scrap.py** (500+ lines)
   - Rename to a01_collect_corp_events.py
   - Split EDGAR logic and splits logic
   - Extract ticker management

### Files to Delete

1. `service/stock_data_manager.py` - Replaced by v2
2. `service/identity_cal.py` - Empty module
3. `service/model_service.py` - Merge into utils.py
4. `b01train_winner_classifier_pct_oof_fix.py` - If not canonical

### Files to Rename

1. `edga_events_scrap.py` → `a01_collect_corp_events.py`
2. `a00build_dataset_with_features.py` → `a03_build_features.py`
3. `service/winner_scoring_refactor.py` → `service/winner_scoring.py`
4. `service/stock_data_manager2.py` → `service/stock_data_manager.py`

---

## 💡 **Design Principles for Refactoring**

1. **Single Responsibility** - Each module should have one clear purpose
2. **DRY (Don't Repeat Yourself)** - Consolidate duplicate code
3. **Explicit Dependencies** - Pass config as parameters, not globals
4. **Fail Fast** - Validate inputs early, raise errors explicitly
5. **Config-Driven** - No hard-coded paths or magic numbers
6. **Testable** - Design for unit and integration testing
7. **Discoverable** - Clear naming makes purpose obvious
8. **Documented** - Architecture and data flows clearly documented

---

## 📝 **Next Steps**

1. Review this plan and prioritize based on your immediate needs
2. Start with Tier 1 fixes (they affect correctness)
3. Create feature branch for refactoring work
4. Tackle one category at a time
5. Write tests as you refactor
6. Update CLAUDE.md to reflect new structure

**Estimated Effort:**
- Tier 1: 2-3 days
- Tier 2: 5-7 days
- Tier 3: 3-5 days
- Total: 2-3 weeks with testing

**Risk Level:** Medium
- Most changes are consolidations (low risk)
- Config changes need careful testing
- Pipeline scripts need validation with real data

---

This pipeline shows good intentions (modular services, configuration-driven, test coverage) but suffers from **refactoring inertia** - multiple valid implementations were created without consolidating best practices. The highest-impact fixes would be:

1. Consolidate winner_scoring modules
2. Fix config fallback logic
3. Validate data at pipeline boundaries
4. Rename/reorganize scripts for clarity
