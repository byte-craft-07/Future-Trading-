# =============================================================
# FILE: trading_system/phase_e/session_signals.py
# PURPOSE: Task 2 — E1 to E30 session-based signals
# =============================================================

import numpy as np
import pandas as pd
from config_e import CONFIG, PHASE_E_SIGNAL_NAMES, ALL_78_SIGNALS


def add_session_signals(df: pd.DataFrame,
                         config: dict) -> pd.DataFrame:
    """
    30 NSE session-based strategy signals add karo.
    Kill zones, sweeps, FVG, institutional patterns.

    Args:
        df    : Phase D DataFrame with all indicators.
        config: CONFIG dict.

    Returns:
        DataFrame with E1-E30 + Master_Score columns.
    """
    print("\n" + "─" * 58)
    print("🕐 ADDING 30 SESSION SIGNALS (Phase E)")
    print("─" * 58)

    # ── Shorthand ──
    c   = df["Close"];    o   = df["Open"]
    h   = df["High"];     l   = df["Low"]
    vol = df["Volume"];   vma = df["Volume_MA_20"]
    rsi = df["RSI_14"]
    vwap= df["VWAP"]  if "VWAP"  in df.columns else df["Close"]
    ema9= df["EMA_9"] if "EMA_9" in df.columns else df["Close"]
    ema21=df["EMA_21"]if "EMA_21"in df.columns else df["Close"]
    pdh = df["PDH"]   if "PDH"   in df.columns else df["High"]
    pdl = df["PDL"]   if "PDL"   in df.columns else df["Low"]
    atr = df["ATR_14"]if "ATR_14"in df.columns else pd.Series(1, index=df.index)
    mso = df["Minutes_Since_Open"] if "Minutes_Since_Open" in df.columns \
          else pd.Series(60, index=df.index)

    # Candle features (fallback if not present)
    bs  = df["Body_Size"]   if "Body_Size"   in df.columns \
          else abs(c - o) / c * 100
    lw  = df["Lower_Wick"]  if "Lower_Wick"  in df.columns \
          else (pd.DataFrame({"o": o, "c": c}).min(axis=1) - l) / c * 100
    cbr = (abs(c - o) / (h - l + 0.0001))   # Candle body ratio

    # ── HELPER: Kill Zone flags ───────────────────────────────
    Morning_KZ   = mso <= 30                  # 09:15 – 09:45
    Midday_Zone  = (mso >= 105) & (mso <= 225)# 11:00 – 13:00
    Afternoon_KZ = mso >= 315                 # 14:30 – 15:30
    In_Any_KZ    = Morning_KZ | Afternoon_KZ

    # ── HELPER: Weekly Levels ─────────────────────────────────
    w_bars        = config["weekly_period"] * 75
    Weekly_High   = h.rolling(w_bars).max().shift(1)
    Weekly_Low    = l.rolling(w_bars).min().shift(1)

    # ── HELPER: Internal Liquidity Levels ─────────────────────
    Internal_High = h.rolling(10).max().shift(1)
    Internal_Low  = l.rolling(10).min().shift(1)

    # ─────────────────────────────────────────────────────────
    # E1 — Kill Zone Sweep Reversal
    # ─────────────────────────────────────────────────────────
    df["E1_KZ_Sweep_Reversal"] = (
        (l < Internal_Low) &
        (c > Internal_Low) &
        In_Any_KZ &
        (c > o)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E2 — Morning Kill Zone Setup
    # ─────────────────────────────────────────────────────────
    sweep_m = l < l.shift(1).rolling(3).min()
    df["E2_Morning_KZ"] = (
        Morning_KZ & sweep_m &
        (c > o) & (c > vwap)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E3 — Afternoon Kill Zone Setup
    # ─────────────────────────────────────────────────────────
    df["E3_Afternoon_KZ"] = (
        Afternoon_KZ &
        (l < Internal_Low) &
        (c > (h + l) / 2) &
        (vol > vma * 1.3)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E4 — PDH Sweep Reversal (SELL)
    # ─────────────────────────────────────────────────────────
    df["E4_PDH_Sweep"] = (
        (h > pdh) &
        (c < pdh) &
        (c < o) &
        (rsi > 65)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E5 — PDL Sweep Reversal (BUY)
    # ─────────────────────────────────────────────────────────
    df["E5_PDL_Sweep"] = (
        (l < pdl) &
        (c > pdl) &
        (c > o) &
        (c > o.shift(1))
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E6 — Weekly High Sweep (SELL)
    # ─────────────────────────────────────────────────────────
    df["E6_Weekly_High_Sweep"] = (
        (h > Weekly_High) &
        (c < Weekly_High) &
        In_Any_KZ
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E7 — Weekly Low Sweep (BUY)
    # ─────────────────────────────────────────────────────────
    df["E7_Weekly_Low_Sweep"] = (
        (l < Weekly_Low) &
        (c > Weekly_Low) &
        In_Any_KZ
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E8 — Sweep and Structure Shift (BOS after sweep)
    # ─────────────────────────────────────────────────────────
    df["E8_Sweep_Shift"] = (
        (l < Internal_Low) &
        (c > Internal_Low) &
        (c > h.shift(1))
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E9 — Double Liquidity Sweep
    # ─────────────────────────────────────────────────────────
    df["E9_Double_Sweep"] = (
        (l.shift(2) < Internal_Low.shift(2)) &
        (l < Internal_Low) &
        (c > Internal_Low)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E10 — Demand Zone Bounce
    # ─────────────────────────────────────────────────────────
    last_bull_low = l.rolling(20).min().shift(5)
    df["E10_Demand_Sweep"] = (
        (l <= last_bull_low * 1.005) &
        (c > last_bull_low) &
        (c > o)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E11 — Supply Zone Rejection (SELL)
    # ─────────────────────────────────────────────────────────
    supply_lvl = h.rolling(20).max().shift(3)
    df["E11_Supply_Sweep"] = (
        (h >= supply_lvl * 0.998) &
        (c < supply_lvl) &
        (c < o)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E12 — Liquidity Pool Trap (sharp spike reversal)
    # ─────────────────────────────────────────────────────────
    df["E12_Liq_Pool_Trap"] = (
        (l < l.shift(1) * 0.998) &
        (c > l * 1.003) &
        (vol > vma * 2.0)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E13 — Stop Hunt Fade (hunt lows then reverse)
    # ─────────────────────────────────────────────────────────
    low10 = l.rolling(10).min().shift(1)
    df["E13_Stop_Hunt"] = (
        (l < low10) &
        (c > low10) &
        (c > (h + l) / 2)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E14 — Fair Value Gap (FVG) Reaction
    # ─────────────────────────────────────────────────────────
    fvg_low  = h.shift(2)              # Bottom of gap
    fvg_high = l                       # Top of gap (current)
    gap_exists = fvg_low < fvg_high
    in_fvg     = (l <= fvg_high) & (h >= fvg_low)
    df["E14_FVG"] = (
        gap_exists & in_fvg & (c > o)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E15 — Reclaim After Sweep (confirmed reversal)
    # ─────────────────────────────────────────────────────────
    df["E15_Reclaim"] = (
        (l.shift(1) < Internal_Low.shift(1)) &
        (c > Internal_Low) &
        (c > o)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E16 — Rejection / Pin Bar Candle at Support
    # ─────────────────────────────────────────────────────────
    df["E16_Rejection_Candle"] = (
        (lw > 0.6) &
        (bs < 0.2) &
        (abs(l - Internal_Low) / (l + 0.001) < 0.003)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E17 — Breakout After Prior Sweep
    # ─────────────────────────────────────────────────────────
    df["E17_Breakout_Sweep"] = (
        (l.shift(3) < Internal_Low.shift(3)) &
        (c.shift(2) > Internal_Low.shift(3)) &
        (c > h.rolling(5).max().shift(1))
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E18 — Session Midpoint Recovery (morning)
    # ─────────────────────────────────────────────────────────
    day_high = df.groupby("Date")["High"].transform("max")
    day_low  = df.groupby("Date")["Low"].transform("min")
    midpoint = (day_high + day_low) / 2
    df["E18_Midpoint"] = (
        (c < midpoint) &
        (c > c.shift(1)) &
        Morning_KZ
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E19 — Institutional Reversal (KZ + sweep + engulf)
    # ─────────────────────────────────────────────────────────
    df["E19_Institutional_Rev"] = (
        In_Any_KZ &
        (l < Internal_Low) &
        (c > o) &
        (c > o.shift(1)) &
        (o < c.shift(1)) &
        (vol > vma * 1.5)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E20 — Institutional Continuation (no rejection)
    # ─────────────────────────────────────────────────────────
    df["E20_Institutional_Cont"] = (
        (lw < 0.2) &
        (cbr > 0.7) &
        (c > c.shift(1)) &
        (c.shift(1) > c.shift(2)) &
        (c > vwap)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E21 — High Quality Confluence (all 5 conditions)
    # ─────────────────────────────────────────────────────────
    df["E21_HQ_Confluence"] = (
        In_Any_KZ &
        (l < Internal_Low) &
        (c > Internal_Low) &
        (c > h.shift(1)) &
        (c > vwap)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E22 — Broken Level Retest (intraday)
    # ─────────────────────────────────────────────────────────
    old_res = h.rolling(15).max().shift(8)
    df["E22_Level_Retest"] = (
        (c.shift(4) > old_res.shift(4)) &
        (abs(c - old_res) / (c + 0.001) < 0.002) &
        (c > o) & (c > c.shift(1))
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E23 — Sweep to Daily Liquidity (PDL bounce)
    # ─────────────────────────────────────────────────────────
    df["E23_Daily_Liq"] = (
        (abs(l - pdl) / (pdl + 0.001) < 0.003) &
        (c > pdl) & (c > o)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E24 — One-Way Session Bias
    # ─────────────────────────────────────────────────────────
    first_close = df.groupby("Date")["Close"].transform("first")
    df["E24_Session_Bias"] = (
        (first_close < c) &
        (c > vwap) &
        (ema9 > ema21)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E25 — No Rejection Continuation (SELL)
    # ─────────────────────────────────────────────────────────
    df["E25_No_Rejection_Cont"] = (
        (l.shift(1) < Internal_Low.shift(1)) &
        (c.shift(1) < Internal_Low.shift(1)) &
        (c < c.shift(1)) &
        (vol > vma)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E26 — Minimum 1:2 RR Filter
    # ─────────────────────────────────────────────────────────
    next_res = h.rolling(10).max().shift(1)
    room     = (next_res - c) >= (atr * 4)
    df["E26_RR_Filter"] = (
        room & (c > vwap)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E27 — Sweep with Order Block Zone
    # ─────────────────────────────────────────────────────────
    bear_ob_high  = o.shift(3)
    bear_ob_low   = c.shift(3)
    bear_ob_valid = o.shift(3) > c.shift(3)
    in_ob = (l <= bear_ob_high) & (h >= bear_ob_low)
    df["E27_OB_Sweep"] = (
        bear_ob_valid & in_ob &
        (l < Internal_Low) &
        (c > bear_ob_low)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E28 — High Volume Reversal
    # ─────────────────────────────────────────────────────────
    df["E28_HV_Reversal"] = (
        (vol > vma * 2.5) &
        (l < Internal_Low) &
        (c > o)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E29 — Post-Sweep Pullback Entry
    # ─────────────────────────────────────────────────────────
    e1_col = "E1_KZ_Sweep_Reversal"
    prior_rev = df[e1_col].shift(5) if e1_col in df.columns \
                else pd.Series(0, index=df.index)
    df["E29_Sweep_Pullback"] = (
        (prior_rev == 1) &
        (abs(c - Internal_Low.shift(5)) / (c + 0.001) < 0.003) &
        (c > o)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # E30 — Expert Session Reversal (most selective)
    # ─────────────────────────────────────────────────────────
    df["E30_Expert_Rev"] = (
        In_Any_KZ &
        (l < Internal_Low) &
        (c > Internal_Low) &
        (c > h.shift(1)) &
        (c > vwap) &
        (rsi > 35) & (rsi < 65) &
        (vol > vma * 1.3)
    ).astype(int)

    # ── Master Confluence Score (0 - 78) ─────────────────────
    avail_all = [s for s in ALL_78_SIGNALS if s in df.columns]
    df["Master_Score"]      = df[avail_all].sum(axis=1)
    df["Session_Strong"]    = (df["Master_Score"] >= 4).astype(int)
    df["Expert_Signal"]     = df["E30_Expert_Rev"]  # Most selective

    # ── Print frequency + win rate ────────────────────────────
    total = len(df)
    n_ahead = 2
    future_up = (df["Close"].shift(-n_ahead) > df["Close"]).astype(int)

    print(f"\n{'Signal':<25} {'Fired':>7} {'Win%':>7}")
    print("─" * 42)
    for col in PHASE_E_SIGNAL_NAMES:
        if col not in df.columns:
            continue
        sig_rows = df[df[col] == 1]
        n  = len(sig_rows)
        wr = (future_up.loc[sig_rows.index].sum() / n * 100
              if n > 0 else 0.0)
        flag = "✅" if wr >= 55 else "  "
        print(f"{flag} {col:<23} {n:>7} {wr:>6.1f}%")

    print(f"\n✅ 30 session signals added")
    print(f"   Master Score max : {df['Master_Score'].max()}")
    print(f"   Total signals    : {len(avail_all)}/78")

    return df

# EXPLANATION: E30 Expert_Rev sabse selective signal
# hai — 7 conditions saath honni chahiye. Agar yeh
# fire kare to bahut high probability trade hai.
# Kill zones mein sabse zyada sweep-and-reverse
# patterns milte hain — institutions wahan orders
# fill karte hain.