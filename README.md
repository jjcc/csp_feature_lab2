
# CSP Feature Lab

This script iterates **cash-secured put** snapshot CSVs, labels expiry outcomes with **yfinance**, and runs a **RandomForest** to rank which parameters are most related to profitability.

## Install
```
pip install -r requirements.txt
```

## Run
```
python analyze_csp_features.py --data_dir /path/to/folder --glob "coveredPut_*.csv"
```

Optional:
- `--max_rows 200` for a quick run

## Inputs: expected columns (missing ones are tolerated)
- baseSymbol, expirationDate, strike, bid, ask, delta, moneyness
- impliedVolatilityRank1y, potentialReturn, potentialReturnAnnual
- breakEvenProbability, percentToBreakEvenBid, openInterest, volume
- tradeTime, underlyingLastPrice

## Outputs
- `labeled_trades.csv` — all rows with computed expiry close, PnL, return_pct, and `win` label
- `feature_importances.csv` — feature rankings
- `classification_report.txt` — train/test metrics
