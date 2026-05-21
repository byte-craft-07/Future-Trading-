# =============================================================
# FILE: trading_system/phase_c/backtest_engine.py
# PURPOSE: Task 3 — Production backtest engine from scratch
# RULES:
#   - Entry at NEXT DAY Open (no look-ahead)
#   - Commission + Slippage both sides
#   - SL/TP checked via daily Low/High
#   - One trade at a time
# =============================================================

import numpy as np
import pandas as pd


class BacktestEngine:
    """
    Custom backtesting engine for NSE daily-candle strategies.

    Features:
      - Realistic entry at next-day Open
      - Stop Loss & Take Profit via ATR
      - Commission + Slippage applied
      - One position at a time
      - Full institutional metrics
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        """
        Initialize engine with data and settings.

        Args:
            df    : Full featured DataFrame (all dates).
            config: CONFIG dict with backtest parameters.
        """
        self.df              = df.reset_index(drop=True)
        self.config          = config
        self.initial_capital = config["initial_capital"]
        self.commission      = config["commission_pct"]
        self.slippage        = config["slippage_pct"]
        self.pos_size        = config["position_size"]
        self.sl_atr          = config["stop_loss_atr"]
        self.tp_atr          = config["take_profit_atr"]

    # ─────────────────────────────────────────────────────────
    def run_strategy(self, signal_column: str) -> dict:
        """
        Ek strategy signal column ke liye full backtest run karo.

        Entry Logic:
          - Signal = 1 on day N
          - Buy at Open of day N+1 + slippage
        Exit Logic:
          - Check daily Low vs Stop Loss
          - Check daily High vs Take Profit
          - Exit at whichever triggers first
        Position Size:
          - 10% of current free capital per trade

        Args:
            signal_column: Column name (e.g. "S1_RSI_MACD_Reversal")

        Returns:
            Dict with trades list, equity curve, metrics.
        """
        df = self.df

        if signal_column not in df.columns:
            return {"error": f"Column {signal_column} not found"}

        n              = len(df)
        free_capital   = float(self.initial_capital)
        equity_curve   = []
        trades         = []

        # Trade state variables
        in_trade     = False
        entry_price  = 0.0
        stop_loss    = 0.0
        take_profit  = 0.0
        entry_date   = None
        entry_idx    = 0
        shares       = 0.0
        prev_signal  = False   # Did yesterday have a signal?

        for i in range(n):
            # ── Step 1: ENTER if prev day had signal ──────────
            if prev_signal and not in_trade and i > 0:
                raw_open    = df["Open"].iloc[i]
                entry_price = raw_open * (1 + self.slippage)

                # Position size = 10% of current capital
                trade_val = free_capital * self.pos_size
                if trade_val < entry_price:
                    prev_signal = False
                else:
                    shares = trade_val / entry_price

                    # Commission on entry
                    entry_comm  = shares * entry_price * self.commission
                    free_capital -= (shares * entry_price + entry_comm)

                    # ATR-based Stop Loss and Take Profit
                    atr         = df["ATR_14"].iloc[i]
                    stop_loss   = entry_price - (atr * self.sl_atr)
                    take_profit = entry_price + (atr * self.tp_atr)

                    entry_date  = df["Date"].iloc[i]
                    entry_idx   = i
                    in_trade    = True

            prev_signal = False  # Reset each iteration

            # ── Step 2: CHECK EXIT if in trade ────────────────
            if in_trade:
                lo = df["Low"].iloc[i]
                hi = df["High"].iloc[i]

                exit_price = None
                exit_type  = None

                # Stop Loss hit — price dropped to SL
                if lo <= stop_loss:
                    exit_price = stop_loss * (1 - self.slippage)
                    exit_type  = "SL"

                # Take Profit hit — price rose to TP
                elif hi >= take_profit:
                    exit_price = take_profit * (1 - self.slippage)
                    exit_type  = "TP"

                # Force exit on last day
                elif i == n - 1:
                    exit_price = df["Close"].iloc[i] * (1 - self.slippage)
                    exit_type  = "EOD"

                if exit_price is not None:
                    exit_val   = shares * exit_price
                    exit_comm  = exit_val * self.commission
                    free_capital += (exit_val - exit_comm)

                    # Trade P&L metrics
                    pct_ret    = (exit_price - entry_price) / entry_price * 100
                    entry_comm_abs = shares * entry_price * self.commission
                    abs_pnl    = ((exit_price - entry_price) * shares
                                  - entry_comm_abs - exit_comm)

                    trades.append({
                        "entry_date" : str(entry_date)[:10],
                        "exit_date"  : str(df["Date"].iloc[i])[:10],
                        "entry_price": round(entry_price, 2),
                        "exit_price" : round(exit_price, 2),
                        "exit_type"  : exit_type,
                        "pct_return" : round(pct_ret, 3),
                        "abs_pnl"    : round(abs_pnl, 2),
                        "win"        : exit_price > entry_price,
                        "hold_days"  : i - entry_idx,
                    })

                    in_trade    = False
                    shares      = 0.0
                    entry_price = 0.0

            # ── Step 3: RECORD equity (end of day) ────────────
            if in_trade:
                # Mark-to-market: free cash + position value
                curr_equity = free_capital + shares * df["Close"].iloc[i]
            else:
                curr_equity = free_capital
            equity_curve.append(curr_equity)

            # ── Step 4: CHECK new signal (for tomorrow) ────────
            if (not in_trade
                    and i < n - 1
                    and df[signal_column].iloc[i] == 1):
                prev_signal = True

        # ── Calculate metrics from trades + equity ──
        dates   = df["Date"].tolist()
        metrics = self.calculate_metrics(trades, equity_curve, dates)
        metrics["signal_column"] = signal_column
        metrics["equity_curve"]  = equity_curve
        metrics["dates"]         = [str(d)[:10] for d in dates]

        return metrics

    # ─────────────────────────────────────────────────────────
    def calculate_metrics(self, trades: list,
                          equity_curve: list,
                          dates: list) -> dict:
        """
        Institutional-grade metrics calculate karo.

        Metrics:
          Basic    : trades, win rate, hold period
          Returns  : total return, buy&hold, avg trade
          Risk     : Sharpe, Max Drawdown, Calmar
          Edge     : Profit Factor, Expectancy

        Args:
            trades      : List of trade dicts.
            equity_curve: Daily portfolio value list.
            dates       : Corresponding dates list.

        Returns:
            Dict with all calculated metrics.
        """
        # ── Handle zero trades case ──
        if not trades:
            return {
                "total_trades": 0, "win_rate": 0,
                "total_return": 0, "sharpe": -99,
                "max_drawdown": 0, "calmar": 0,
                "profit_factor": 0, "expectancy": 0,
            }

        equity = pd.Series(equity_curve,
                           index=pd.to_datetime(dates))

        # ── Basic counts ──
        n_trades = len(trades)
        wins     = [t for t in trades if t["win"]]
        losses   = [t for t in trades if not t["win"]]
        win_rate = len(wins) / n_trades * 100

        # ── Return metrics ──
        total_ret = ((equity.iloc[-1] - self.initial_capital)
                     / self.initial_capital * 100)

        # Buy & Hold: just hold stock entire period
        bnh = ((self.df["Close"].iloc[-1] - self.df["Close"].iloc[0])
               / self.df["Close"].iloc[0] * 100)

        rets = [t["pct_return"] for t in trades]
        avg_ret   = float(np.mean(rets))
        best_ret  = float(np.max(rets))
        worst_ret = float(np.min(rets))
        avg_hold  = float(np.mean([t["hold_days"] for t in trades]))

        # ── Sharpe Ratio ──
        daily_ret  = equity.pct_change().dropna()
        rf_daily   = 0.065 / 252             # India risk-free ~6.5%
        if daily_ret.std() > 1e-10:
            sharpe = ((daily_ret.mean() - rf_daily)
                      / daily_ret.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        # ── Max Drawdown ──
        running_max = equity.cummax()
        drawdown    = (equity - running_max) / running_max * 100
        max_dd      = float(drawdown.min())

        # ── Calmar Ratio ──
        # Annual return / |Max Drawdown|
        years        = len(equity) / 252
        annual_ret   = total_ret / years if years > 0 else 0
        calmar       = (annual_ret / abs(max_dd)
                        if max_dd != 0 else 0.0)

        # ── Profit Factor ──
        gross_wins   = sum(t["abs_pnl"] for t in wins) if wins else 0
        gross_losses = abs(sum(t["abs_pnl"] for t in losses)) if losses else 1e-9
        pf           = gross_wins / gross_losses

        # ── Expectancy (₹ per trade) ──
        avg_win_r  = np.mean([t["abs_pnl"] for t in wins])  if wins   else 0
        avg_loss_r = np.mean([t["abs_pnl"] for t in losses]) if losses else 0
        win_r_rate = len(wins)   / n_trades
        los_r_rate = len(losses) / n_trades
        expectancy = (win_r_rate * avg_win_r) - (los_r_rate * abs(avg_loss_r))

        return {
            "total_trades"    : n_trades,
            "win_trades"      : len(wins),
            "loss_trades"     : len(losses),
            "win_rate"        : round(win_rate, 2),
            "total_return"    : round(total_ret, 2),
            "bnh_return"      : round(bnh, 2),
            "avg_trade_return": round(avg_ret, 3),
            "best_trade"      : round(best_ret, 3),
            "worst_trade"     : round(worst_ret, 3),
            "avg_hold_days"   : round(avg_hold, 1),
            "sharpe"          : round(float(sharpe), 3),
            "max_drawdown"    : round(max_dd, 2),
            "calmar"          : round(float(calmar), 3),
            "profit_factor"   : round(pf, 3),
            "expectancy"      : round(float(expectancy), 2),
            "trades"          : trades,
        }

# EXPLANATION: Yeh engine realistic conditions
# simulate karta hai:
# - Slippage = price thoda unfavorable hota hai entry/exit pe
# - Commission = broker fee dono side
# - ATR-based SL/TP = market volatility ke hisaab se
# - Sharpe > 1 = good, > 2 = excellent
# - MaxDD < 20% = safe for real money
# - Expectancy > 0 = strategy positive edge hai