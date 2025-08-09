
# CSP Portfolio Simulator

Event-driven simulator for **cash-secured puts** using  `labeled_trades.csv`.

## How it works
- **Entry** at `tradeTime`: reserve capital = `strike*100*contracts`
- **Exit** at `expirationDate`: realize PnL = `entry_credit - exit_intrinsic` (per contract), minus commissions
- Manages **overlapping positions**, **capital allocation**, and **max concurrent**

## Run
```bash
python simulate_csp_portfolio.py \
  --csv labeled_trades.csv \
  --initial_cash 100000 \
  --per_trade_capital_pct 0.1 \
  --max_concurrent 5 \
  --commission_per_contract 0.65 \
  --sort_by breakEvenProbability \
  --sort_desc 1
```

### Key knobs
- `--per_trade_capital_pct`: cap per position as % of equity (e.g., 0.1 = 10%)
- `--max_concurrent`: limit number of open positions
- `--sort_by`: if too many candidates on a day, pick the highest (e.g., set `breakEvenProbability`)
- `--no_compounding`: size off initial cash instead of current equity
- `--contracts_per_trade`: scale up position size

## Outputs
- `equity_curve.csv` — timestamped equity and cash
- `portfolio_trades.csv` — realized trades with PnL and reserved capital
- `portfolio_summary.json` — final metrics

## Tips
- Pre-filter `labeled_trades.csv` with  chosen **risk filters** (e.g., BEP≥0.78, PBE≤-6%, moneyness≤-4%, delta≤0.3, IVR≤50–60) to minimize tail losses.
- You can also run multiple passes with different `--sort_by` (e.g., `percentToBreakEvenBid` or `potentialReturnAnnual`) to test selection heuristics.
