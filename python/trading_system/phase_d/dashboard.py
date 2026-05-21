# =============================================================
# FILE: trading_system/phase_d/dashboard.py
# PURPOSE: Task 6 — 4-panel dark theme live dashboard
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from config_d import CONFIG, PLOTS_DIR_D, INTRADAY_SIGNALS


def _draw_candlesticks(ax, df_plot: pd.DataFrame):
    """
    Manual candlestick drawing on a matplotlib axis.
    Green = bullish candle, Red = bearish candle.
    """
    bull_color = "#26a69a"   # Teal green
    bear_color = "#ef5350"   # Red

    for i, (_, row) in enumerate(df_plot.iterrows()):
        color = bull_color if row["Close"] >= row["Open"] else bear_color

        # Wick (thin line high to low)
        ax.plot([i, i], [row["Low"], row["High"]],
                color=color, linewidth=0.7, zorder=2)

        # Body (rectangle)
        body_bot = min(row["Open"], row["Close"])
        body_h   = abs(row["Close"] - row["Open"])
        if body_h < 0.01:
            body_h = 0.01    # Minimum visible body
        rect = mpatches.Rectangle(
            (i - 0.35, body_bot), 0.7, body_h,
            color=color, zorder=3
        )
        ax.add_patch(rect)


def build_live_dashboard(df: pd.DataFrame,
                          signal: dict,
                          config: dict,
                          save: bool = True):
    """
    4-panel professional trading dashboard.
    Panel 1: Price + VWAP + EMA + ORB + PDH + signals
    Panel 2: RSI
    Panel 3: MACD
    Panel 4: Volume

    Args:
        df    : Full intraday DataFrame.
        signal: Latest signal dict from signal_engine.
        config: CONFIG dict.
        save  : Whether to save PNG.

    Returns:
        fig, axes tuple.
    """
    # ── Slice last N candles ──
    n        = config["display_candles"]
    df_plot  = df.tail(n).reset_index(drop=True)

    # ── Dark theme setup ──
    plt.style.use("dark_background")
    bg  = "#1e1e2e"
    fg  = "#cdd6f4"

    fig = plt.figure(figsize=(16, 12), facecolor=bg)
    gs  = gridspec.GridSpec(
        4, 2,
        height_ratios=[3, 1, 1, 1],
        hspace=0.08, wspace=0.15
    )

    ax1 = fig.add_subplot(gs[0, :])    # Price (full width)
    ax2 = fig.add_subplot(gs[1, 0])    # RSI
    ax3 = fig.add_subplot(gs[1, 1])    # MACD
    ax4 = fig.add_subplot(gs[2:, :])   # Volume (full width)

    x   = np.arange(len(df_plot))

    # ═══════════════════════════════════════
    # PANEL 1 — PRICE CHART
    # ═══════════════════════════════════════
    ax1.set_facecolor(bg)

    # Draw candlesticks
    _draw_candlesticks(ax1, df_plot)

    # ── VWAP line ──
    ax1.plot(x, df_plot["VWAP"], color="#89b4fa",
             linewidth=1.5, label="VWAP", zorder=4)

    # ── EMA lines ──
    if "EMA_9" in df_plot:
        ax1.plot(x, df_plot["EMA_9"],  color="#f38ba8",
                 linewidth=1.0, label="EMA 9",  alpha=0.8)
    if "EMA_21" in df_plot:
        ax1.plot(x, df_plot["EMA_21"], color="#fab387",
                 linewidth=1.0, label="EMA 21", alpha=0.8)

    # ── ORB levels ──
    if "ORB_High" in df_plot:
        ax1.axhline(df_plot["ORB_High"].iloc[-1],
                    color="#a6e3a1", linewidth=1.0,
                    linestyle="--", label="ORB High", alpha=0.7)
        ax1.axhline(df_plot["ORB_Low"].iloc[-1],
                    color="#f38ba8", linewidth=1.0,
                    linestyle="--", label="ORB Low", alpha=0.7)

    # ── PDH / PDL ──
    if "PDH" in df_plot:
        ax1.axhline(df_plot["PDH"].iloc[-1],
                    color="#cba6f7", linewidth=0.8,
                    linestyle=":", label="PDH", alpha=0.7)
        ax1.axhline(df_plot["PDL"].iloc[-1],
                    color="#cba6f7", linewidth=0.8,
                    linestyle=":", label="PDL", alpha=0.7)

    # ── BUY signals (green triangles) ──
    buy_rows = df_plot[df_plot["Total_Score"] >= 5]
    if len(buy_rows) > 0:
        ax1.scatter(buy_rows.index,
                    buy_rows["Low"] * 0.998,
                    marker="^", color="#a6e3a1",
                    s=80, zorder=6, label="BUY Signal")

    # ── SELL signals (red triangles) — D7 or D20 ──
    sell_cols = [c for c in ["D7_HTF_Trap", "D20_News_Reject"]
                 if c in df_plot.columns]
    if sell_cols:
        sell_rows = df_plot[df_plot[sell_cols].sum(axis=1) >= 1]
        if len(sell_rows) > 0:
            ax1.scatter(sell_rows.index,
                        sell_rows["High"] * 1.002,
                        marker="v", color="#f38ba8",
                        s=80, zorder=6, label="SELL Signal")

    # ── Predicted price star (10 candles ahead) ──
    pred_px = signal.get("predicted_price", None)
    if pred_px:
        pred_x = len(df_plot) + 2   # 2 candles ahead
        ax1.plot([len(df_plot)-1, pred_x],
                 [df_plot["Close"].iloc[-1], pred_px],
                 color="#f9e2af", linewidth=1.0,
                 linestyle=":", alpha=0.8)
        ax1.scatter(pred_x, pred_px,
                    marker="*", color="#f9e2af",
                    s=200, zorder=7, label=f"Pred ₹{pred_px:.1f}")

    ax1.set_xlim(-1, len(df_plot) + 5)
    ax1.set_ylabel("Price (₹)", color=fg, fontsize=9)
    ax1.tick_params(colors=fg, labelsize=7)
    ax1.legend(loc="upper left", fontsize=7,
               facecolor=bg, edgecolor="gray",
               labelcolor=fg, ncol=3)
    ax1.grid(True, alpha=0.1, color="gray")

    # ═══════════════════════════════════════
    # PANEL 2 — RSI
    # ═══════════════════════════════════════
    ax2.set_facecolor(bg)
    ax2.plot(x, df_plot["RSI_14"], color="#89dceb",
             linewidth=1.0, label="RSI 14")
    ax2.axhline(70, color="#f38ba8", linewidth=0.8,
                linestyle="--", alpha=0.7)
    ax2.axhline(30, color="#a6e3a1", linewidth=0.8,
                linestyle="--", alpha=0.7)
    ax2.fill_between(x, 70, df_plot["RSI_14"],
                     where=df_plot["RSI_14"] >= 70,
                     color="#f38ba8", alpha=0.15)
    ax2.fill_between(x, 30, df_plot["RSI_14"],
                     where=df_plot["RSI_14"] <= 30,
                     color="#a6e3a1", alpha=0.15)
    curr_rsi = df_plot["RSI_14"].iloc[-1]
    ax2.set_ylim(0, 100)
    ax2.set_ylabel(f"RSI {curr_rsi:.1f}", color=fg, fontsize=8)
    ax2.tick_params(colors=fg, labelsize=7)
    ax2.grid(True, alpha=0.1)

    # ═══════════════════════════════════════
    # PANEL 3 — MACD
    # ═══════════════════════════════════════
    ax3.set_facecolor(bg)
    ax3.plot(x, df_plot["MACD"],        color="#89b4fa",
             linewidth=1.0, label="MACD")
    ax3.plot(x, df_plot["MACD_Signal"], color="#fab387",
             linewidth=1.0, label="Signal")

    # Histogram bars
    colors_hist = ["#a6e3a1" if v >= 0 else "#f38ba8"
                   for v in df_plot["MACD_Hist"]]
    ax3.bar(x, df_plot["MACD_Hist"],
            color=colors_hist, alpha=0.6, width=0.6)
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.set_ylabel("MACD", color=fg, fontsize=8)
    ax3.tick_params(colors=fg, labelsize=7)
    ax3.grid(True, alpha=0.1)

    # ═══════════════════════════════════════
    # PANEL 4 — VOLUME
    # ═══════════════════════════════════════
    ax4.set_facecolor(bg)
    vol_colors = ["#a6e3a1" if r["Close"] >= r["Open"]
                  else "#f38ba8"
                  for _, r in df_plot.iterrows()]
    ax4.bar(x, df_plot["Volume"],
            color=vol_colors, alpha=0.7, width=0.7)
    if "Volume_MA_20" in df_plot:
        ax4.plot(x, df_plot["Volume_MA_20"],
                 color="#f9e2af", linewidth=1.0,
                 label="Vol MA20", alpha=0.8)

    # Highlight volume spikes
    vma_vals = df_plot["Volume_MA_20"]
    spikes   = df_plot[df_plot["Volume"] > vma_vals * 2.0]
    ax4.scatter(spikes.index, spikes["Volume"],
                color="white", s=20, zorder=5,
                label="Spike")
    ax4.set_ylabel("Volume", color=fg, fontsize=8)
    ax4.tick_params(colors=fg, labelsize=7)
    ax4.grid(True, alpha=0.1)
    ax4.legend(loc="upper left", fontsize=7,
               facecolor=bg, labelcolor=fg)

    # ── Main title ──
    sig_str  = signal.get("signal", "NEUTRAL")
    pred_p   = signal.get("predicted_price", 0)
    conf     = signal.get("confidence", 50)
    fig.suptitle(
        f"RELIANCE.NS — LIVE  |  Signal: {sig_str}  "
        f"|  Predicted: ₹{pred_p:,.2f}  "
        f"|  Confidence: {conf:.1f}%",
        color=fg, fontsize=11, fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        p = PLOTS_DIR_D / "live_dashboard.png"
        plt.savefig(p, dpi=150, bbox_inches="tight",
                    facecolor=bg)
        print(f"📊 Dashboard saved: {p.name}")

    return fig

# EXPLANATION: Dark theme trading terminal look.
# Panel 1 mein sab kuch dikhta hai — price, VWAP,
# ORB levels, signal markers. Predicted price star
# aage 10 min ka point dikhaata hai. Ye screenshot
# leke WhatsApp pe share bhi kar sakte hain!