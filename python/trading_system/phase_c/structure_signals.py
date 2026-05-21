# =============================================================
# FILE: trading_system/phase_c/structure_signals.py
# PURPOSE: Task 2 — C1 to C8 structure-based signals
# RULE: shift(1) use karo — no look-ahead bias
# =============================================================

import pandas as pd
import numpy as np
from config_c import CONFIG, PHASE_B_STRATEGIES, PHASE_C_STRATEGIES


def add_structure_signals(df: pd.DataFrame,
                          config: dict) -> pd.DataFrame:
    """
    8 structure-based strategy signals add karo.
    Ye price structure (breakouts, BOS, divergence) track karte hain.

    Args:
        df    : Phase B DataFrame with all indicators.
        config: CONFIG dict with thresholds.

    Returns:
        DataFrame with C1-C8 columns + updated Confluence.
    """
    print("\n" + "─" * 58)
    print("🏗️  ADDING 8 STRUCTURE SIGNALS (Phase C)")
    print("─" * 58)

    c   = df["Close"]
    o   = df["Open"]
    h   = df["High"]
    l   = df["Low"]
    vol = df["Volume"]
    vma = df["Volume_MA_20"]
    rsi = df["RSI_14"]
    sma20 = df["SMA_20"]

    hlr = df["High_Low_Range"]   # (H-L)/C*100

    sr  = config["sr_period"]    # 20

    # ─────────────────────────────────────────────────────────
    # C1 — False Breakout Reversal
    # Price broke resistance yesterday, came back today
    # ─────────────────────────────────────────────────────────
    resistance     = h.rolling(sr).max().shift(1)
    broke_above    = c.shift(1) > resistance.shift(1)  # Prev broke
    back_below     = c < resistance                     # Today below
    vol_confirm    = vol > vma

    df["C1_False_Breakout"] = (
        broke_above & back_below & vol_confirm
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # C2 — False Trendline Break (Recovery)
    # Price dipped below trendline but reclaimed it
    # ─────────────────────────────────────────────────────────
    tl_slope   = l.rolling(sr).min()                   # Trendline proxy
    below_tl   = l.shift(1) < tl_slope.shift(1)       # Prev below
    reclaimed  = c > tl_slope                          # Today above
    bull_body  = c > o

    df["C2_False_Trendline"] = (
        below_tl & reclaimed & bull_body
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # C3 — Break of Structure (BOS)
    # Was making lower highs, now broke above swing high
    # ─────────────────────────────────────────────────────────
    swing_high  = h.rolling(config["bos_swing_period"]).max().shift(1)
    bos_bullish = c > swing_high                        # Broke above
    prev_lh     = h.shift(1) < h.shift(2)              # Was lower high
    vol_strong  = vol > vma * 1.2

    df["C3_BOS"] = (
        bos_bullish & prev_lh & vol_strong
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # C4 — First Pullback After Breakout
    # Recent breakout → small pullback → now resuming
    # ─────────────────────────────────────────────────────────
    recent_high = h.rolling(20).max().shift(1)
    broke_out   = h.shift(3) > recent_high.shift(3)    # Broke 3 days ago
    above_ma    = c > sma20
    below_bo    = c < h.shift(3)                        # Below breakout pt
    small_candle= hlr < 1.0                             # Consolidation
    break_again = c > h.shift(1)                        # Resuming up

    df["C4_First_Pullback"] = (
        broke_out & above_ma & below_bo &
        small_candle & break_again
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # C5 — Retest of Broken Resistance (Now Support)
    # Old resistance → price broke it → now retesting as support
    # ─────────────────────────────────────────────────────────
    old_res     = h.rolling(20).max().shift(5)          # 5 days ago level
    was_above   = c.shift(3) > old_res.shift(3)         # Was above it
    near_level  = abs(c - old_res) / (c + 0.001) < 0.015  # Within 1.5%
    bouncing    = (c > o) & (c > c.shift(1))

    df["C5_Retest"] = (
        was_above & near_level & bouncing
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # C6 — Range Expansion Breakout
    # After tight consolidation → explosive big candle
    # ─────────────────────────────────────────────────────────
    avg_range = hlr.rolling(10).mean()
    tight      = hlr.shift(1) < (avg_range.shift(1) *
                                  config["range_tight_pct"])
    expanded   = hlr > avg_range * 2.0                  # Big candle today
    bull_vol   = (c > o) & (vol > vma * 1.5)

    df["C6_Range_Expansion"] = (
        tight & expanded & bull_vol
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # C7 — RSI Bullish Divergence
    # Price made lower low BUT RSI made higher low
    # ─────────────────────────────────────────────────────────
    price_ll   = c < c.rolling(5).min().shift(1)        # Price lower low
    rsi_hl     = rsi > rsi.rolling(5).min().shift(1)    # RSI higher low
    rsi_zone   = (rsi > 25) & (rsi < 50)               # Recovery zone
    bull_candle= c > o

    df["C7_RSI_Divergence"] = (
        price_ll & rsi_hl & rsi_zone & bull_candle
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # C8 — Market Structure Shift
    # Was in downtrend (lower highs) → now broke above swing
    # ─────────────────────────────────────────────────────────
    ll1         = h.shift(4) > h.shift(3)               # Lower high 2
    ll2         = h.shift(3) > h.shift(2)               # Lower high 1
    was_down    = ll1 & ll2                              # Was downtrend
    broke_swing = c > h.rolling(5).max().shift(1)       # Structure break
    vol_spike   = vol > vma * 1.3

    df["C8_Structure_Shift"] = (
        was_down & broke_swing & vol_spike
    ).astype(int)

    # ── Update Confluence Score (S1-S15 + C1-C8 = max 23) ───
    all_sig_cols = PHASE_B_STRATEGIES + PHASE_C_STRATEGIES
    available    = [c for c in all_sig_cols if c in df.columns]
    df["Confluence_Score"]    = df[available].sum(axis=1)
    df["Confluence_Strong_C"] = (
        df["Confluence_Score"] >= config["min_confluence"]
    ).astype(int)

    # ── Print frequency report ────────────────────────────────
    total = len(df)
    print(f"\n{'Signal':<25} {'Fired':>7} {'% Days':>8}")
    print("─" * 43)
    for col in PHASE_C_STRATEGIES:
        if col in df.columns:
            n = df[col].sum()
            print(f"{col:<25} {n:>7} {n/total*100:>7.1f}%")

    print(f"\nUpdated Confluence Score  — max: "
          f"{df['Confluence_Score'].max():.0f}")
    print(f"Days with Score >= {config['min_confluence']}  "
          f"— count: {df['Confluence_Strong_C'].sum()}")

    return df

# EXPLANATION: Structure signals zyada reliable hote
# hain kyunki ye price behavior directly track karte
# hain — breakouts, retests, divergence. Inka combination
# Phase B ke momentum signals ke saath powerful hai.
# Ab total 23 signals hain model ke paas.