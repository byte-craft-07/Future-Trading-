# Trading System

Python-based multi-phase trading research system for data preparation, feature engineering, model training, backtesting, live signal generation, and dashboard outputs.

## Project Layout

- `phase_a/` - baseline data loading, cleaning, features, model utilities, and visualizations
- `phase_b/` - strategy rules, indicators, confluence logic, and model reporting
- `phase_c/` - structure signals, backtesting engine, and phase C model flow
- `phase_d/` - intraday/live pipeline, signal engine, dashboard, and prediction helpers
- `phase_e/` - ensemble, sentiment, LSTM models, final dashboard, and session signals

## GitHub Safety

This repository intentionally ignores generated/private artifacts such as:

- raw and processed market data
- trained model files (`.pkl`, `.keras`, etc.)
- generated plots and dashboards
- logs, live signals, and backtest outputs
- local `.env` files and editor settings

Before pushing publicly, add any required API keys or broker credentials through environment variables only. Do not commit real credentials.

## Basic Usage

Run a phase entrypoint from its phase directory, for example:

```bash
python phase_a/phase_a_main.py
```

Install project dependencies according to the imports used by the phase you are running, such as `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and deep learning libraries for Phase E.
