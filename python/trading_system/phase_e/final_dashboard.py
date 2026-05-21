# =============================================================
# FILE: trading_system/phase_e/final_dashboard.py
# PURPOSE: Task 7 — Complete final system dashboard (6 panels)
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from config_e import CONFIG, PLOTS_DIR_E
from session_signals import PHASE_E_SIGNAL_NAMES


def build_final_dashboard(df: pd.DataFrame,
                           signal: dict,
                           config: dict):
    """
    6-panel complete trading dashboard.
    Panel 1: Price + VWAP + EMA + ORB + signals
    Panel 2: RSI
    Panel 3: MACD
    Panel 4: Volume
    Panel 5: Master Confluence Score
    Panel 6: Signal Component Breakdown

    Args:
        df    : Full intraday DataFrame.
        signal: Ensemble signal dict.
        config: CONFIG dict.
    """
    n        = config.get("display_candles", 100)
    df_p     = df.tail(n).reset_index(drop=True)
    x        = np.arange(len(df_p))

    # ── Dark theme ──
    plt.style.use("dark_background")
    bg, fg = "#1e1e2e", "#cdd6f4"

    fig = plt.figure(figsize=(18, 14), facecolor=bg)
    gs  = gridspec.GridSpec(
        4, 2,
        height_ratios=[3, 1, 1, 1],
        hspace=0.1, wspace=0.15
    )

    ax1 = fig.add_subplot(gs[0, :])   # Price (full width)
    ax2 = fig.add_subplot(gs[1, 0])   # RSI
    ax3 = fig.add_subplot(gs[1, 1])   # MACD
    ax4 = fig.add_subplot(gs[2, :])   # Volume
    ax5 = fig.add_subplot(gs[3, 0])   # Master Score
    ax6 = fig.add_subplot(gs[3, 1])   # Ensemble breakdown

    for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
        ax.set_facecolor(bg)

    bull_c, bear_c = "#26a69a", "#ef5350"

    # ════════════════════════════════
    # PANEL 1 — PRICE
    # ════════════════════════════════
    for i, (_, row) in enumerate(df_p.iterrows()):
        cc = bull_c if row["Close"] >= row["Open"] else bear_c
        ax1.plot([i, i], [row["Low"], row["High"]],
                 color=cc, linewidth=0.7)
        bbot = min(row["Open"], row["Close"])
        bh   = max(abs(row["Close"] - row["Open"]), 0.01)
        ax1.add_patch(mpatches.Rectangle(
            (i-0.35, bbot), 0.7, bh, color=cc))

    # VWAP
    if "VWAP" in df_p:
        ax1.plot(x, df_p["VWAP"],  color="#89b4fa",
                 linewidth=1.5, label="VWAP")
    if "EMA_9" in df_p:
        ax1.plot(x, df_p["EMA_9"], color="#f38ba8",
                 linewidth=1.0, label="EMA9", alpha=0.8)
    if "EMA_21" in df_p:
        ax1.plot(x, df_p["EMA_21"],color="#fab387",
                 linewidth=1.0, label="EMA21", alpha=0.8)
    # ORB / PDH / PDL
    for col, col2, lbl in [
        ("ORB_High","","ORB H"), ("ORB_Low","","ORB L"),
        ("PDH","","PDH"), ("PDL","","PDL")
    ]:
        if col in df_p:
            ax1.axhline(df_p[col].iloc[-1],
                        color="#cba6f7", linewidth=0.7,
                        linestyle=":", alpha=0.6, label=lbl)

    # Expert signal markers (E30)
    if "E30_Expert_Rev" in df_p:
        exp = df_p[df_p["E30_Expert_Rev"] == 1]
        ax1.scatter(exp.index, exp["Low"] * 0.997,
                    marker="*", color="gold",
                    s=120, zorder=7, label="⭐E30")

    # Master score >= 8 → strong signal
    if "Master_Score" in df_p:
        strong = df_p[df_p["Master_Score"] >= 8]
        ax1.scatter(strong.index, strong["Low"] * 0.999,
                    marker="^", color="#a6e3a1",
                    s=80, zorder=6, label="Strong(8+)")

    # Predicted price
    pred_p = signal.get("predicted_price")
    if pred_p:
        ax1.plot([len(df_p)-1, len(df_p)+2],
                 [df_p["Close"].iloc[-1], pred_p],
                 color="#f9e2af", linewidth=1.2,
                 linestyle=":", alpha=0.9)
        ax1.scatter(len(df_p)+2, pred_p, marker="D",
                    color="#f9e2af", s=100, zorder=8,
                    label=f"Pred ₹{pred_p:.0f}")

    ax1.set_xlim(-1, len(df_p)+6)
    ax1.set_ylabel("Price (₹)", color=fg, fontsize=9)
    ax1.legend(loc="upper left", fontsize=7, facecolor=bg,
               labelcolor=fg, ncol=4)
    ax1.grid(True, alpha=0.08)

    # ════════════════════════════════
    # PANEL 2 — RSI
    # ════════════════════════════════
    if "RSI_14" in df_p:
        ax2.plot(x, df_p["RSI_14"], color="#89dceb", linewidth=0.9)
        ax2.axhline(70, color="#f38ba8", linewidth=0.7, linestyle="--")
        ax2.axhline(30, color="#a6e3a1", linewidth=0.7, linestyle="--")
        ax2.fill_between(x, 70, df_p["RSI_14"],
                         where=df_p["RSI_14"]>=70,
                         color="#f38ba8", alpha=0.12)
        ax2.fill_between(x, 30, df_p["RSI_14"],
                         where=df_p["RSI_14"]<=30,
                         color="#a6e3a1", alpha=0.12)
        rsi_now = df_p["RSI_14"].iloc[-1]
        ax2.set_ylim(0, 100)
        ax2.set_ylabel(f"RSI {rsi_now:.1f}", color=fg, fontsize=8)
    ax2.grid(True, alpha=0.08)

    # ════════════════════════════════
    # PANEL 3 — MACD
    # ════════════════════════════════
    if "MACD" in df_p and "MACD_Signal" in df_p:
        ax3.plot(x, df_p["MACD"],        color="#89b4fa", linewidth=0.9)
        ax3.plot(x, df_p["MACD_Signal"], color="#fab387", linewidth=0.9)
        hist = df_p["MACD_Hist"]
        clrs = ["#a6e3a1" if v >= 0 else "#f38ba8" for v in hist]
        ax3.bar(x, hist, color=clrs, alpha=0.6, width=0.6)
        ax3.axhline(0, color="gray", linewidth=0.4)
    ax3.set_ylabel("MACD", color=fg, fontsize=8)
    ax3.grid(True, alpha=0.08)

    # ════════════════════════════════
    # PANEL 4 — VOLUME
    # ════════════════════════════════
    vcols = [bull_c if r["Close"]>=r["Open"] else bear_c
             for _, r in df_p.iterrows()]
    ax4.bar(x, df_p["Volume"], color=vcols, alpha=0.7, width=0.7)
    if "Volume_MA_20" in df_p:
        ax4.plot(x, df_p["Volume_MA_20"],
                 color="#f9e2af", linewidth=0.9, alpha=0.8)
    ax4.set_ylabel("Volume", color=fg, fontsize=8)
    ax4.grid(True, alpha=0.08)

    # ════════════════════════════════
    # PANEL 5 — Master Score Heatmap
    # ════════════════════════════════
    if "Master_Score" in df_p:
        ms_vals = df_p["Master_Score"].values
        ms_cols = plt.cm.RdYlGn(ms_vals / 78)
        ax5.bar(x, ms_vals, color=ms_cols, alpha=0.8, width=0.8)
        ax5.axhline(8, color="gold", linewidth=1.0,
                    linestyle="--", label="Strong (8)")
        ax5.set_ylabel("Master\nScore", color=fg, fontsize=8)
        ax5.set_ylim(0, max(ms_vals.max()+2, 12))
        ax5.legend(fontsize=7, facecolor=bg, labelcolor=fg)
    ax5.grid(True, alpha=0.08)

    # ════════════════════════════════
    # PANEL 6 — Ensemble Breakdown
    # ════════════════════════════════
    components = {
        "ML (35%)":    signal.get("ml_probability", 50),
        "LSTM (35%)":  signal.get("lstm_price_prob", 50),
        "Vol  (15%)":  signal.get("vol_confidence", 50),
        "Sent (15%)":  (signal.get("sentiment_score", 0)+1)/2*100,
    }
    bars_c = ["#89b4fa","#a6e3a1","#fab387","#cba6f7"]
    ax6.barh(list(components.keys()),
             list(components.values()),
             color=bars_c, alpha=0.8)
    ax6.axvline(50, color="gray", linewidth=0.8, linestyle="--")
    ax6.axvline(signal.get("ensemble_confidence", 50),
                color="gold", linewidth=1.5,
                linestyle="-", label=f"Ensemble "
                f"{signal.get('ensemble_confidence',50):.1f}%")
    ax6.set_xlim(0, 100)
    ax6.set_xlabel("Bullish %", color=fg, fontsize=8)
    ax6.set_title("Model Ensemble Breakdown",
                  color=fg, fontsize=8, pad=3)
    ax6.legend(fontsize=7, facecolor=bg, labelcolor=fg)
    ax6.grid(True, axis="x", alpha=0.08)
    for ax in [ax2,ax3,ax4,ax5,ax6]:
        ax.tick_params(colors=fg, labelsize=7)

    # ── Main title ──
    ens   = signal.get("ensemble_confidence", 50)
    sig_s = signal.get("signal", "NEUTRAL")
    ms    = signal.get("master_score", 0)
    e30   = "⭐E30!" if signal.get("expert_signal") else ""
    fig.suptitle(
        f"RELIANCE.NS — FINAL SYSTEM  |  {sig_s}  "
        f"|  Pred ₹{signal.get('predicted_price',0):,.0f}  "
        f"|  Ensemble {ens:.1f}%  |  Score {ms}/78  {e30}",
        color=fg, fontsize=11, fontweight="bold", y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    p = PLOTS_DIR_E / "final_dashboard.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close()
    print(f"📊 Final dashboard saved: {p.name}")
    return p

# EXPLANATION: 6-panel dashboard mein sab kuch
# ek jagah dikhta hai. Panel 6 ensemble breakdown
# dikhata hai — kaunse model ne kitna contribute
# kiya. Gold star (E30) = expert signal active hai.