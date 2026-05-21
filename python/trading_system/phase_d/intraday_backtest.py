# =============================================================
# FILE: trading_system/phase_d/intraday_backtest.py
# PURPOSE: Task 8 — Quick win rate check on 5min signals
# =============================================================

import pandas as pd
from config_d import INTRADAY_SIGNALS


def quick_backtest_intraday(df: pd.DataFrame,
                             config: dict) -> pd.DataFrame:
    """
    Har D-signal ke liye win rate calculate karo.
    Win = next 2 candles mein Close > current Close.

    Args:
        df    : Full intraday DataFrame.
        config: CONFIG dict.

    Returns:
        DataFrame with signal win rates.
    """
    print("\n" + "═" * 55)
    print("⚡ INTRADAY SIGNAL WIN RATE ANALYSIS")
    print("═" * 55)

    n_ahead = config["predict_candles"]    # 2 candles = 10 min

    # Future close (no leakage — only for evaluation)
    future_close = df["Close"].shift(-n_ahead)
    went_up      = (future_close > df["Close"]).astype(int)

    rows = []
    for sig in INTRADAY_SIGNALS:
        if sig not in df.columns:
            continue

        signal_rows = df[df[sig] == 1]
        count = len(signal_rows)
        if count == 0:
            win_rate = 0.0
        else:
            wins     = went_up.loc[signal_rows.index].sum()
            win_rate = wins / count * 100

        rows.append({
            "Signal"  : sig,
            "Count"   : count,
            "Win Rate": round(win_rate, 1)
        })

    result_df = pd.DataFrame(rows).sort_values(
        "Win Rate", ascending=False
    ).reset_index(drop=True)

    # ── Print table ──
    print(f"\n{'#':>3}  {'Signal':<25} {'Count':>7} {'Win Rate':>9}")
    print("─" * 48)
    for i, row in result_df.iterrows():
        flag  = "✅" if row["Win Rate"] >= 55 else "⚠️ "
        print(f"{flag} {i+1:>2}.  {row['Signal']:<25} "
              f"{int(row['Count']):>7} {row['Win Rate']:>8.1f}%")

    # ── Best signal ──
    if len(result_df) > 0:
        best = result_df.iloc[0]
        print(f"\n🏆 Best Intraday Signal: {best['Signal']}")
        print(f"   Win Rate : {best['Win Rate']}%  "
              f"({int(best['Count'])} signals)")

    print("\nℹ️  Win Rate > 55% = signal adds edge on 5min data")

    return result_df

# EXPLANATION: Intraday backtest se pata chalega
# kaunse D-signals 5min pe kaam karte hain.
# 50% = random guess. 60%+ = genuine edge.
# Top signals ko live system mein highlight karenge.