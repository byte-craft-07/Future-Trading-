# Trading System

This folder contains the main Python trading research system used by the `Future-Trading-` repository. The system is organized into five phases, starting from baseline feature generation and ending with an ensemble signal engine.

## Project Layout

- `phase_a/` - data loading, cleaning, feature engineering, target creation, and baseline ML model
- `phase_b/` - strategy indicators, rule-based signals, confluence score, and model reporting
- `phase_c/` - market structure signals, backtesting engine, reports, and strategy ranking
- `phase_d/` - intraday pipeline, prediction helpers, signal engine, and live-style dashboard
- `phase_e/` - ensemble engine, sentiment input, LSTM models, session signals, and final dashboard

## Run Order

Run phases from the `python/trading_system` directory:

```bash
python phase_a/phase_a_main.py
python phase_b/phase_b_main.py
python phase_c/phase_c_main.py
python phase_d/phase_d_main.py
python phase_e/phase_e_main.py
```

Later phases depend on outputs from earlier phases, so run them in order when rebuilding the complete system.

## Local Outputs

This repository intentionally ignores generated/private artifacts such as:

- raw and processed market data
- trained model files (`.pkl`, `.keras`, etc.)
- generated plots and dashboards
- logs, live signals, and backtest outputs
- local `.env` files and editor settings

These files are created locally when the system runs, but they should not be committed to GitHub.

## Configuration

Each phase has its own config file:

- `phase_a/config.py`
- `phase_b/config_b.py`
- `phase_c/config_c.py`
- `phase_d/config_d.py`
- `phase_e/config_e.py`

Use these files for non-secret settings such as ticker, interval, model parameters, thresholds, and output paths. Keep API keys, broker credentials, access tokens, and account details outside the repository.

## Disclaimer

This project is for research and learning. It is not financial advice. Validate every strategy independently before any real-market use.
