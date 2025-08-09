
# ML Trade Selector + Simulator

This pipeline trains a model to pick **higher-probability CSP sells** and then runs a **portfolio simulation** on the selected trades.

## Install
```
pip install -r requirements_ml.txt
```

## Run
```
python train_csp_model_and_sim.py \
  --csv labeled_trades.csv \
  --initial_cash 100000 \
  --per_trade_capital_pct 0.30 \
  --daily_risk_cap_pct 1.00 \
  --max_concurrent 10 \
  --commission_per_contract 0.0 \
  --target_coverage 0.20 \
  --model gradient_boosting \
  --save_prefix mlgb
```

## Outputs
- `ml_metrics.json` — ROC-AUC, AP, top-k precision & avg returns
- `threshold_table.csv` — coverage vs threshold table on test set
- `selected_trades.csv` — trades picked by the model on full data
- `equity_curve_<prefix>.csv`, `portfolio_trades_<prefix>.csv`, `portfolio_summary_<prefix>.json` — sim results
