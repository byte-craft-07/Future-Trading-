# =============================================================
# FILE: trading_system/phase_b/confluence.py
# PURPOSE: Task 4 — Strategy Confluence Score banao + plot
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
from config_b import PLOTS_DIR_B

# All 15 strategy column names
STRATEGY_COLS = [
    "S1_RSI_MACD_Reversal", "S2_SMA_Crossover",
    "S3_RSI_Pullback",      "S4_MACD_Momentum",
    "S5_Volume_Breakout",   "S6_Engulf_MACD",
    "S7_Oversold_Bounce",   "S8_Trend_Follow",
    "S9_Momentum_Breakout", "S10_EMA_Cross",
    "S11_Gap_Up",           "S12_Support_Bounce",
    "S13_Pullback_Entry",   "S14_Tiny_Candle",
    "S15_Supertrend"
]


def add_confluence_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Confluence Score = kitni strategies ek saath agree kar
    rahi hain (0 to 15). 3+ = strong signal.

    Args:
        df: DataFrame with S1-S15 columns.

    Returns:
        DataFrame with Confluence_Score + Confluence_Strong.
    """
    print("\n" + "─" * 55)
    print("🔗 CONFLUENCE SCORE ANALYSIS")
    print("─" * 55)

    # ── 4a: Sum of all 15 strategy signals ───────────────────
    df["Confluence_Score"] = df[STRATEGY_COLS].sum(axis=1)

    # ── 4b: Strong signal = 3 or more strategies agree ───────
    df["Confluence_Strong"] = (df["Confluence_Score"] >= 3).astype(int)

    # ── 4c: Analysis by target class ─────────────────────────
    up_days   = df[df["Target"] == 1]
    down_days = df[df["Target"] == 0]
    strong_pct= df["Confluence_Strong"].mean() * 100

    print(f"Avg Confluence on UP   days : {up_days['Confluence_Score'].mean():.3f}")
    print(f"Avg Confluence on DOWN days : {down_days['Confluence_Score'].mean():.3f}")
    print(f"Days with Score >= 3        : {strong_pct:.1f}%")

    # ── 4d: Plot histogram ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    # UP days = green, DOWN days = red
    ax.hist(up_days["Confluence_Score"],   bins=range(17),
            alpha=0.6, color="green", label="UP days (Target=1)",
            align="left")
    ax.hist(down_days["Confluence_Score"], bins=range(17),
            alpha=0.6, color="red",   label="DOWN days (Target=0)",
            align="left")

    ax.axvline(x=3, color="blue", linestyle="--", linewidth=1.5,
               label="Strong Signal (≥3)")
    ax.set_title("Strategy Confluence Score Distribution", fontsize=13,
                 fontweight="bold")
    ax.set_xlabel("Number of Strategies Agreeing (0-15)")
    ax.set_ylabel("Frequency (Days)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(16))

    plt.tight_layout()
    path = PLOTS_DIR_B / "confluence_score.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Chart saved: {path}")

    return df

# EXPLANATION: Confluence = agreement. Jab sirf 1
# strategy signal deti hai → weak. Jab 5-6 strategies
# ek saath BUY bol rahi hain → bahut strong. Model yeh
# score feature ke roop mein use karega. UP days pe
# average score DOWN days se zyada hona chahiye —
# tab ye feature useful hai.