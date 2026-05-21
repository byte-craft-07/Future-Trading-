# =============================================================
# FILE: trading_system/phase_a/visualizer.py
# PURPOSE: Task 10 — Price chart + RSI subplot banao
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates    # Date formatting x-axis pe

from config import PLOTS_DIR


def plot_price_chart(df: pd.DataFrame, config: dict):
    """
    Close price + SMA_20 + SMA_50 chart banao.
    RSI subplot neeche add karo.
    Combined PNG save karo plots/ mein.

    Args:
        df    : Featured DataFrame (SMA aur RSI columns hone chahiye).
        config: CONFIG dict (ticker name title ke liye).
    """
    print("\n" + "─" * 50)
    print("📉 GENERATING PRICE CHART")
    print("─" * 50)

    # Date column datetime type mein convert karo
    df["Date"] = pd.to_datetime(df["Date"])

    # ── 2 subplots: price (top) + RSI (bottom) ──
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},  # Price 3x height
        sharex=True    # Dono ka x-axis (Date) same
    )

    # ── Subplot 1: Price + Moving Averages ──
    ax1.plot(df["Date"], df["Close"],
             color="#1f77b4", linewidth=1.2,
             label="Close Price", alpha=0.9)
    ax1.plot(df["Date"], df["SMA_20"],
             color="#ff7f0e", linewidth=1.5,
             label="SMA 20", linestyle="--")
    ax1.plot(df["Date"], df["SMA_50"],
             color="#2ca02c", linewidth=1.5,
             label="SMA 50", linestyle="-.")

    ax1.set_title(f"{config['ticker']} — Price with Moving Averages",
                  fontsize=14, fontweight="bold", pad=10)
    ax1.set_ylabel("Price (₹)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # ── Subplot 2: RSI ──
    ax2.plot(df["Date"], df["RSI_14"],
             color="#9467bd", linewidth=1.2, label="RSI 14")

    # Overbought / Oversold lines
    ax2.axhline(70, color="red",   linewidth=1,
                linestyle="--", alpha=0.7, label="Overbought (70)")
    ax2.axhline(30, color="green", linewidth=1,
                linestyle="--", alpha=0.7, label="Oversold (30)")

    # Zones shade karo
    ax2.fill_between(df["Date"], 70, df["RSI_14"],
                     where=df["RSI_14"] >= 70, alpha=0.15, color="red")
    ax2.fill_between(df["Date"], 30, df["RSI_14"],
                     where=df["RSI_14"] <= 30, alpha=0.15, color="green")

    ax2.set_ylabel("RSI", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle="--")

    # X-axis date formatting
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()

    # ── Save chart ──
    chart_path = PLOTS_DIR / "price_with_MA.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Chart saved: {chart_path}")