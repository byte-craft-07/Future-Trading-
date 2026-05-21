# =============================================================
# FILE: trading_system/phase_b/visualizer_b.py
# PURPOSE: Task 5 — Strategy signal charts banao
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config_b import PLOTS_DIR_B
from confluence import STRATEGY_COLS


def plot_strategy_signals(df: pd.DataFrame):
    """
    Price chart pe strong BUY signals (Confluence>=3) dikhaao.
    Strategy frequency bar chart bhi banao.

    Args:
        df: DataFrame with Close, Date, Confluence columns.
    """
    print("\n" + "─" * 55)
    print("📊 PLOTTING STRATEGY SIGNALS")
    print("─" * 55)

    df["Date"] = pd.to_datetime(df["Date"])

    # ── 5a: Price chart with BUY signal markers ───────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    # Close price line
    ax.plot(df["Date"], df["Close"],
            color="#1f77b4", linewidth=1.0,
            label="Close Price", alpha=0.8)

    # Green triangles where Confluence >= 3 (strong BUY)
    strong_signals = df[df["Confluence_Score"] >= 3]
    ax.scatter(strong_signals["Date"], strong_signals["Close"],
               marker="^", color="lime", s=60, zorder=5,
               label=f"Strong BUY Signal (≥3 strategies)\n"
                     f"[{len(strong_signals)} signals]",
               alpha=0.85, edgecolors="green", linewidths=0.5)

    ax.set_title("RELIANCE.NS — Strategy Confluence BUY Signals",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    p1 = PLOTS_DIR_B / "strategy_signals.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Signal chart saved  : {p1}")

    # ── 5b: Strategy frequency horizontal bar chart ───────────
    fig, ax = plt.subplots(figsize=(10, 7))

    # Count how many times each strategy fired
    signal_counts = {col: df[col].sum() for col in STRATEGY_COLS}
    sorted_items  = sorted(signal_counts.items(), key=lambda x: x[1])
    labels        = [k.replace("_", " ") for k, v in sorted_items]
    values        = [v for k, v in sorted_items]

    # Color gradient (more signals = darker)
    colors = plt.cm.RdYlGn(
        [v / max(values) for v in values]
    )

    bars = ax.barh(labels, values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_title("Strategy Signal Frequency (Days Fired)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Days Signal = 1")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    p2 = PLOTS_DIR_B / "strategy_frequency.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Frequency chart saved: {p2}")

# EXPLANATION: Green triangles dekhoge price chart
# pe — ye woh din hain jab 3+ strategies ek saath
# BUY signal de rahi thi. Agar in points ke baad
# price upar gayi consistently → strategies kaam
# kar rahi hain.