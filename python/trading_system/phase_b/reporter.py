# =============================================================
# FILE: trading_system/phase_b/reporter.py
# PURPOSE: Task 9 — Strategy win rate report banao
# =============================================================

import pandas as pd
from confluence import STRATEGY_COLS


def generate_strategy_report(df: pd.DataFrame):
    """
    Har strategy ke liye report:
      - Kitni baar signal fire hua
      - Win rate (jab signal=1, Target=1 ka %)

    Args:
        df: DataFrame with strategy columns + Target.

    Prints:
        Formatted table sorted by win rate.
    """
    print("\n" + "═" * 55)
    print("📋 STRATEGY WIN RATE REPORT")
    print("═" * 55)
    print(f"{'Strategy':<25} {'Signals':>8} {'Win Rate':>10}")
    print("─" * 45)

    report_rows = []

    for col in STRATEGY_COLS:
        # Rows where this strategy fired
        signal_rows = df[df[col] == 1]
        n_signals   = len(signal_rows)

        if n_signals == 0:
            win_rate = 0.0
        else:
            # Win rate = % of signal days where price went UP
            win_rate = signal_rows["Target"].mean() * 100

        report_rows.append({
            "strategy" : col,
            "signals"  : n_signals,
            "win_rate" : win_rate
        })

    # Sort by win rate (best first)
    report_rows.sort(key=lambda x: x["win_rate"], reverse=True)

    for row in report_rows:
        # ✅ if win rate > 55%, else ⚠️
        flag = "✅" if row["win_rate"] > 55 else "⚠️ "
        print(f"{flag} {row['strategy']:<23} "
              f"{row['signals']:>8} "
              f"{row['win_rate']:>9.1f}%")

    # ── Top 3 strategies ──
    print("\n" + "─" * 45)
    print("🏆 TOP 3 STRATEGIES BY WIN RATE:")
    for i, row in enumerate(report_rows[:3], 1):
        print(f"  {i}. {row['strategy']:<25} "
              f"Win Rate: {row['win_rate']:.1f}%  "
              f"({row['signals']} signals)")

    print("\nℹ️  Win Rate > 55% = strategy adds real edge.")
    print("   These strategies get PRIORITY in Phase C backtesting.")

# EXPLANATION: Win rate = "jab ye strategy signal
# deti hai, kitne % cases mein price actually upar
# gayi?" Random chance = 50%. 60%+ win rate matlab
# strategy genuinely kuch meaningful pakad rahi hai.
# Top 3 strategies Phase C mein priority signal honge.