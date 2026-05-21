# =============================================================
# FILE: trading_system/phase_c/backtest_runner.py
# PURPOSE: Task 4,5,6,7 — Run all strategies, plots, ranking
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from backtest_engine import BacktestEngine
from config_c import (CONFIG, ALL_STRATEGIES,
                      BT_PLOTS_DIR, BT_RESULTS_DIR)


# =============================================================
# TASK 4 — BACKTEST ALL STRATEGIES
# =============================================================

def backtest_all_strategies(df: pd.DataFrame,
                            config: dict) -> dict:
    """
    Har strategy signal column ke liye backtest run karo.
    Including Confluence_Strong_C (all strategies combined).

    Args:
        df    : Full featured DataFrame.
        config: CONFIG dict.

    Returns:
        Dict mapping strategy_name → metrics dict.
    """
    print("\n" + "─" * 58)
    print("🔄 BACKTESTING ALL 23 STRATEGIES + CONFLUENCE")
    print("─" * 58)

    engine  = BacktestEngine(df, config)
    results = {}

    # All 23 individual strategies
    strategies_to_test = [s for s in ALL_STRATEGIES
                          if s in df.columns]

    # Add confluence strategy
    if "Confluence_Strong_C" in df.columns:
        strategies_to_test.append("Confluence_Strong_C")

    total = len(strategies_to_test)

    for idx, strat in enumerate(strategies_to_test, 1):
        print(f"  [{idx:>2}/{total}] Backtesting {strat}...", end=" ")
        result = engine.run_strategy(strat)
        results[strat] = result

        # Quick summary on same line
        if result.get("total_trades", 0) > 0:
            print(f"✅  Trades={result['total_trades']:>3}  "
                  f"Win={result['win_rate']:>5.1f}%  "
                  f"Return={result['total_return']:>+7.2f}%  "
                  f"Sharpe={result['sharpe']:>6.3f}")
        else:
            print("⚠️  No trades generated")

    print(f"\n✅ Backtesting complete — {len(results)} strategies tested.")
    return results


# =============================================================
# TASK 5 — EQUITY CURVE PLOTS
# =============================================================

def plot_equity_curves(results: dict, df: pd.DataFrame):
    """
    Top 5 strategies ki equity curves plot karo.
    Buy & Hold baseline bhi dikhao.

    Args:
        results: Dict of strategy → metrics.
        df     : DataFrame (for dates + BnH reference).
    """
    print("\n" + "─" * 58)
    print("📈 PLOTTING EQUITY CURVES")
    print("─" * 58)

    # ── Sort by total return, get top 5 ──
    valid = {k: v for k, v in results.items()
             if v.get("total_trades", 0) > 0}
    if not valid:
        print("⚠️  No valid results to plot")
        return

    top5 = sorted(valid.items(),
                  key=lambda x: x[1]["total_return"],
                  reverse=True)[:5]

    # ── Buy & Hold equity curve ──
    init_cap   = CONFIG["initial_capital"]
    bnh_equity = (df["Close"] / df["Close"].iloc[0]) * init_cap
    dates_dt   = pd.to_datetime(df["Date"])

    # ── Plot top 5 ──
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ["#2ecc71", "#3498db", "#e67e22",
              "#9b59b6", "#1abc9c"]

    for (name, metrics), color in zip(top5, colors):
        ec = metrics.get("equity_curve", [])
        if len(ec) == len(dates_dt):
            ax.plot(dates_dt, ec, linewidth=1.5,
                    label=f"{name} ({metrics['total_return']:+.1f}%)",
                    color=color)

    # Buy & Hold line
    ax.plot(dates_dt, bnh_equity,
            color="gray", linewidth=1.2,
            linestyle="--",
            label=f"Buy & Hold ({bnh_equity.iloc[-1]/init_cap*100-100:+.1f}%)",
            alpha=0.7)

    ax.axhline(y=init_cap, color="black", linewidth=0.8,
               linestyle=":", alpha=0.5, label="Initial Capital")

    ax.set_title("Top 5 Strategies — Equity Curves (2018-2024)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    p = BT_PLOTS_DIR / "equity_curves_top5.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"✅ Top 5 equity curves: {p.name}")

    # ── Confluence equity curve (separate) ──
    if "Confluence_Strong_C" in results:
        _plot_single_equity(
            results["Confluence_Strong_C"],
            dates_dt, bnh_equity,
            "Confluence Strategy (Score >= 4)",
            "confluence_equity.png"
        )

    # ── Worst strategy (lowest return) ──
    worst_name, worst_metrics = sorted(
        valid.items(),
        key=lambda x: x[1]["total_return"]
    )[0]
    _plot_single_equity(
        worst_metrics, dates_dt, bnh_equity,
        f"Worst: {worst_name}",
        "worst_strategy.png"
    )


def _plot_single_equity(metrics: dict,
                        dates_dt, bnh_equity,
                        title: str, filename: str):
    """Helper: Plot single strategy equity curve."""
    ec = metrics.get("equity_curve", [])
    if not ec or len(ec) != len(dates_dt):
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates_dt, ec, color="#2ecc71", linewidth=1.5,
            label=f"Strategy ({metrics['total_return']:+.1f}%)")
    ax.plot(dates_dt, bnh_equity, color="gray",
            linestyle="--", linewidth=1.2, label="Buy & Hold")
    ax.axhline(y=CONFIG["initial_capital"], color="black",
               linewidth=0.8, linestyle=":")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = BT_PLOTS_DIR / filename
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"✅ Chart saved: {p.name}")


# =============================================================
# TASK 6 — DRAWDOWN ANALYSIS
# =============================================================

def plot_drawdown(results: dict, df: pd.DataFrame):
    """
    Top 3 strategies + Confluence ke liye drawdown charts.

    Args:
        results: Dict of strategy → metrics.
        df     : DataFrame for dates.
    """
    print("\n" + "─" * 58)
    print("📉 DRAWDOWN ANALYSIS")
    print("─" * 58)

    dates_dt = pd.to_datetime(df["Date"])

    valid    = {k: v for k, v in results.items()
                if v.get("total_trades", 0) > 0}

    # Top 3 + confluence
    top3  = sorted(valid.items(),
                   key=lambda x: x[1]["sharpe"],
                   reverse=True)[:3]
    to_plot = top3

    # Add confluence if available
    if "Confluence_Strong_C" in results:
        to_plot = to_plot + [("Confluence_Strong_C",
                               results["Confluence_Strong_C"])]

    for name, metrics in to_plot:
        ec = metrics.get("equity_curve", [])
        if not ec or len(ec) != len(dates_dt):
            continue

        eq      = pd.Series(ec, index=dates_dt)
        peak    = eq.cummax()
        dd      = (eq - peak) / peak * 100         # Drawdown in %

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(dates_dt, dd, 0,
                        color="red", alpha=0.4,
                        label=f"Drawdown (Max={dd.min():.1f}%)")
        ax.axhline(-10, color="orange", linewidth=1,
                   linestyle="--", label="-10% Warning")
        ax.axhline(-20, color="red", linewidth=1,
                   linestyle="--", label="-20% Danger")

        ax.set_title(f"Drawdown Chart — {name}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown %")
        ax.set_ylim(min(dd.min() * 1.1, -25), 2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_name = name.replace("/", "_")
        p = BT_PLOTS_DIR / f"drawdown_{safe_name}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"✅ Drawdown chart: {p.name}  "
              f"(Max DD = {dd.min():.2f}%)")


# =============================================================
# TASK 7 — STRATEGY RANKING TABLE
# =============================================================

def create_ranking_table(results: dict,
                         df: pd.DataFrame) -> pd.DataFrame:
    """
    Sab strategies ko Sharpe Ratio se rank karo.
    Formatted table print karo + bar chart save karo.

    Args:
        results: Dict of strategy → metrics.
        df     : DataFrame (for Buy & Hold calc).

    Returns:
        Ranked DataFrame of all strategy metrics.
    """
    print("\n" + "═" * 58)
    print("🏆 STRATEGY RANKING TABLE (Ranked by Sharpe)")
    print("═" * 58)

    rows = []
    for name, m in results.items():
        if m.get("total_trades", 0) > 0:
            rows.append({
                "Strategy"    : name,
                "Trades"      : m["total_trades"],
                "Win Rate"    : m["win_rate"],
                "Return %"    : m["total_return"],
                "Sharpe"      : m["sharpe"],
                "Max DD %"    : m["max_drawdown"],
                "Profit Factor": m["profit_factor"],
                "Expectancy ₹": m["expectancy"],
            })

    if not rows:
        print("⚠️  No valid results to rank")
        return pd.DataFrame()

    rank_df = pd.DataFrame(rows).sort_values(
        "Sharpe", ascending=False
    ).reset_index(drop=True)
    rank_df.index += 1  # Rank starts at 1

    # ── Print formatted table ──
    hdr = (f"{'Rank':>4}  {'Strategy':<26} {'Trades':>6}  "
           f"{'WinRate':>7}  {'Return%':>8}  "
           f"{'Sharpe':>7}  {'MaxDD%':>7}")
    print(hdr)
    print("─" * 75)

    for rank, row in rank_df.iterrows():
        flag = "🟢" if row["Sharpe"] > 1.0 else "🔴"
        print(f"{flag} {rank:>3}.  "
              f"{row['Strategy']:<26} "
              f"{int(row['Trades']):>6}  "
              f"{row['Win Rate']:>6.1f}%  "
              f"{row['Return %']:>+7.2f}%  "
              f"{row['Sharpe']:>7.3f}  "
              f"{row['Max DD %']:>6.2f}%")

    # ── Special highlights ──
    bnh = ((df["Close"].iloc[-1] - df["Close"].iloc[0])
           / df["Close"].iloc[0] * 100)
    beating = rank_df[rank_df["Return %"] > bnh]

    print(f"\n{'─'*58}")
    print(f"📊 RELIANCE Buy & Hold (2018-2024): {bnh:+.2f}%")
    print(f"   Strategies beating Buy & Hold  : "
          f"{len(beating)}/{len(rank_df)}")
    print(f"\n🏅 BEST Sharpe    : {rank_df.iloc[0]['Strategy']}"
          f" (Sharpe={rank_df.iloc[0]['Sharpe']:.3f})")
    safest = rank_df.loc[rank_df['Max DD %'].idxmax()]
    print(f"🛡️  SAFEST (Min DD): {safest['Strategy']}"
          f" (DD={safest['Max DD %']:.2f}%)")
    active = rank_df.loc[rank_df['Trades'].idxmax()]
    print(f"⚡ MOST ACTIVE    : {active['Strategy']}"
          f" ({int(active['Trades'])} trades)")
    best_wr = rank_df.loc[rank_df['Win Rate'].idxmax()]
    print(f"🎯 BEST WIN RATE  : {best_wr['Strategy']}"
          f" ({best_wr['Win Rate']:.1f}%)")

    # ── Bar chart: Sharpe ranking ──
    _plot_sharpe_ranking(rank_df)

    return rank_df


def _plot_sharpe_ranking(rank_df: pd.DataFrame):
    """Horizontal bar chart — strategies by Sharpe Ratio."""
    plot_df = rank_df.head(20)  # Top 20

    colors = ["seagreen" if s > 1.0 else "tomato"
              for s in plot_df["Sharpe"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        plot_df["Strategy"][::-1],
        plot_df["Sharpe"][::-1],
        color=colors[::-1], edgecolor="white"
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.axvline(x=1.0, color="orange", linewidth=1.5,
               linestyle="--", label="Sharpe = 1.0 (Good)")
    ax.axvline(x=2.0, color="green", linewidth=1.5,
               linestyle="--", label="Sharpe = 2.0 (Excellent)")

    ax.set_title("Strategy Ranking by Sharpe Ratio\n"
                 "(Green = Sharpe > 1, Red = Sharpe < 1)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Sharpe Ratio")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    p = BT_PLOTS_DIR / "strategy_ranking.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"\n📊 Ranking chart saved: {p.name}")

# EXPLANATION: Sharpe Ratio sabse important metric hai.
# 0-1 = average strategy, 1-2 = good, 2+ = excellent.
# MaxDrawdown < 20% = safe for real money.
# Profit Factor > 1.5 = strategy positive edge hai.
# Expectancy > 0 = profitable on average per trade.