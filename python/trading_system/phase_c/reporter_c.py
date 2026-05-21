# =============================================================
# FILE: trading_system/phase_c/reporter_c.py
# PURPOSE: Task 9 — Complete backtest report + Phase D blueprint
# =============================================================

import json
import pandas as pd
from config_c import BT_RESULTS_DIR


def generate_full_report(results: dict,
                         rank_df: pd.DataFrame,
                         df: pd.DataFrame,
                         config: dict):
    """
    Complete Phase C backtest summary report generate karo.
    Top 5 Phase D strategies identify karo.
    Losing strategies flag karo.

    Args:
        results : Dict of strategy → metrics.
        rank_df : Sorted ranking DataFrame.
        df      : Full DataFrame (for B&H).
        config  : CONFIG dict.

    Returns:
        top_strategies list, avoid_strategies list
    """
    print("\n" + "╔" + "═"*54 + "╗")
    print("║" + "     PHASE C BACKTEST REPORT".center(54) + "║")
    print("║" + "     RELIANCE.NS (2018-2024)".center(54) + "║")
    print("╠" + "═"*54 + "╣")

    valid  = {k: v for k, v in results.items()
              if v.get("total_trades", 0) > 0}
    n_test = len(results)

    # ── Best strategy ──
    if len(rank_df) > 0:
        best_row = rank_df.iloc[0]
        best_name   = best_row["Strategy"]
        best_sharpe = best_row["Sharpe"]
        best_ret    = best_row["Return %"]
    else:
        best_name = best_sharpe = best_ret = "N/A"

    # ── Confluence ──
    conf_ret = results.get("Confluence_Strong_C",
                           {}).get("total_return", 0)

    # ── Buy & Hold ──
    bnh = ((df["Close"].iloc[-1] - df["Close"].iloc[0])
           / df["Close"].iloc[0] * 100)

    beating = (rank_df["Return %"] > bnh).sum() if len(rank_df) else 0

    print(f"║  Total Strategies Tested : {n_test:<26}║")
    print(f"║  Best Strategy           : {str(best_name):<26}║")
    print(f"║  Best Sharpe Ratio       : {float(best_sharpe):<26.3f}║")
    print(f"║  Best Return             : {float(best_ret):+26.2f}%║")
    print(f"║  Confluence Return       : {float(conf_ret):+26.2f}%║")
    print(f"║  Buy & Hold Return       : {float(bnh):+26.2f}%║")
    print(f"║  Strategies Beating BnH  : {str(beating)+'/'+str(n_test):<26}║")
    print("╚" + "═"*54 + "╝")

    # ── Top 5 for Phase D ──
    print("\n🚀 TOP 5 STRATEGIES FOR PHASE D (Real-Time):")
    print("─" * 50)
    top5 = rank_df.head(5)
    top_strategies = []
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"  {i}. {row['Strategy']:<28} "
              f"Sharpe={row['Sharpe']:.3f}  "
              f"WinRate={row['Win Rate']:.1f}%")
        top_strategies.append(row["Strategy"])

    # ── Avoid list ──
    print("\n⚠️  STRATEGIES TO AVOID (Sharpe < 0 or WinRate < 40%):")
    print("─" * 50)
    avoid_mask = (rank_df["Sharpe"] < 0) | (rank_df["Win Rate"] < 40)
    avoid_df   = rank_df[avoid_mask]
    avoid_list = []
    if len(avoid_df) > 0:
        for _, row in avoid_df.iterrows():
            print(f"  ❌ {row['Strategy']:<28} "
                  f"Sharpe={row['Sharpe']:.3f}")
            avoid_list.append(row["Strategy"])
    else:
        print("  ✅ No strategies to avoid! All decent.")

    # ── Save JSON report ──
    report = {
        "phase"              : "C",
        "ticker"             : "RELIANCE.NS",
        "period"             : "2018-2024",
        "strategies_tested"  : n_test,
        "best_strategy"      : str(best_name),
        "best_sharpe"        : float(best_sharpe) if best_sharpe != "N/A" else 0,
        "best_return_pct"    : float(best_ret) if best_ret != "N/A" else 0,
        "confluence_return"  : float(conf_ret),
        "buy_and_hold_return": round(float(bnh), 2),
        "strategies_beating_bnh": int(beating),
        "top_5_for_phase_d"  : top_strategies,
        "avoid_list"         : avoid_list,
    }

    try:
        path = BT_RESULTS_DIR / "full_report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Report saved: {path.name}")
    except Exception as e:
        print(f"⚠️  Could not save report JSON: {e}")

    return top_strategies, avoid_list

# EXPLANATION: Yeh Phase D ka blueprint hai.
# Top 5 strategies = wo signals jo real-time
# data pe apply honge. Avoid list = jo backtesting
# mein fail hue — unhe Phase D mein use nahi karenge.
# Confluence beating Buy & Hold = hum simple
# stock hold se zyada kama sakte hain.