# =============================================================
# FILE: trading_system/phase_d/signals_d.py
# PURPOSE: Task 3 — D1 to D25 intraday strategy signals
# RULE: shift() use karo prev candle ke liye — no leakage
# =============================================================

import pandas as pd
from config_d import CONFIG, INTRADAY_SIGNALS


def add_intraday_signals(df: pd.DataFrame,
                          config: dict) -> pd.DataFrame:
    """
    25 intraday strategy signals add karo.
    Har signal = 1 (active) ya 0 (inactive).

    Args:
        df    : DataFrame with all indicators.
        config: CONFIG dict.

    Returns:
        DataFrame with D1-D25 + Total_Score columns.
    """
    print("\n" + "─" * 55)
    print("📡 ADDING 25 INTRADAY SIGNALS")
    print("─" * 55)

    c   = df["Close"]
    o   = df["Open"]
    h   = df["High"]
    l   = df["Low"]
    vol = df["Volume"]
    vma = df["Volume_MA_20"]
    rsi = df["RSI_14"]
    vwap= df["VWAP"]
    vdev= df["VWAP_Dev"]
    ema9= df["EMA_9"]
    ema21=df["EMA_21"]
    macd= df["MACD"]
    msig= df["MACD_Signal"]
    mhst= df["MACD_Hist"]
    orb_h=df["ORB_High"]
    orb_l=df["ORB_Low"]
    pdh = df["PDH"]
    pdl = df["PDL"]
    pdc = df["PDC"]
    atr = df["ATR_14"]
    mso = df["Minutes_Since_Open"]
    bs  = df["Body_Size"]
    uw  = df["Upper_Wick"]

    ob  = config["orb_buffer"]

    # ─────────────────────────────────────────────────────
    # D1 — Opening Range Breakout
    # ─────────────────────────────────────────────────────
    df["D1_ORB_Breakout"] = (
        (mso >= 30) &
        (c > orb_h * (1 + ob)) &
        (vol > vma * 1.3)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D2 — ORB Retest (price pullback to ORB level)
    # ─────────────────────────────────────────────────────
    df["D2_ORB_Retest"] = (
        (c.shift(1) > orb_h) &
        (abs(c - orb_h) / c < 0.003) &
        (c > o)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D3 — VWAP Pullback (touched VWAP, bounced)
    # ─────────────────────────────────────────────────────
    df["D3_VWAP_Pullback"] = (
        (c > vwap) &
        (l <= vwap * 1.001) &
        (c > vwap) &
        (ema9 > ema21)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D4 — VWAP Mean Reversion (too far below → recovery)
    # ─────────────────────────────────────────────────────
    df["D4_VWAP_Reversion"] = (
        (vdev < -config["vwap_dev_pct"]) &
        (c > c.shift(1)) &
        (rsi < 40)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D5 — VWAP Trend Follow (3 candles above VWAP)
    # ─────────────────────────────────────────────────────
    df["D5_VWAP_Trend"] = (
        (c > vwap) &
        (c.shift(1) > vwap.shift(1)) &
        (c.shift(2) > vwap.shift(2)) &
        (c > c.shift(1)) &
        (c.shift(1) > c.shift(2))
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D6 — HTF Bias Continuation (pullback to EMA9)
    # ─────────────────────────────────────────────────────
    df["D6_HTF_Bias"] = (
        (c > pdc) &
        (c > vwap) &
        (ema9 > ema21) &
        (l <= ema9 * 1.002) &
        (c > ema9)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D7 — HTF Trap (near PDH, rejection → reversal)
    # ─────────────────────────────────────────────────────
    df["D7_HTF_Trap"] = (
        (abs(h - pdh) / pdh < 0.005) &
        (c < o) &
        (c < pdh) &
        (rsi > 65)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D8 — Multi-TF BOS (above VWAP + swing break)
    # ─────────────────────────────────────────────────────
    swing_h = h.rolling(10).max().shift(1)
    df["D8_MTF_BOS"] = (
        (c > swing_h) &
        (vol > vma * 1.4) &
        (c > vwap)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D9 — Order Block Reaction
    # ─────────────────────────────────────────────────────
    big_bear = (c.shift(3) < o.shift(3)) & (bs.shift(3) > 0.3)
    in_ob    = (l <= c.shift(3)) & (h >= o.shift(3))
    df["D9_Order_Block"] = (
        big_bear & in_ob &
        (c > o) & (rsi < 55)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D10 — Inducement (fake dip, quick recovery)
    # ─────────────────────────────────────────────────────
    df["D10_Inducement"] = (
        (l < l.shift(1) * 0.999) &
        (c > c.shift(1)) &
        (vol > vma) &
        (c > vwap)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D11 — BOS After Liquidity Sweep
    # ─────────────────────────────────────────────────────
    low5 = l.rolling(5).min().shift(1)
    df["D11_BOS_Sweep"] = (
        (l < low5) &
        (c > low5) &
        (c > h.shift(1))
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D12 — MACD + RSI + VWAP Confluence
    # ─────────────────────────────────────────────────────
    df["D12_MACD_RSI"] = (
        (macd > msig) &
        (mhst > mhst.shift(1)) &
        (rsi > 40) & (rsi < 65) &
        (c > vwap)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D13 — Trendline Trap (dipped below, reclaimed)
    # ─────────────────────────────────────────────────────
    roll_min = l.rolling(20).min()
    df["D13_TL_Trap"] = (
        (l.shift(1) < roll_min.shift(1)) &
        (c > roll_min) &
        (c > o)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D14 — Range Expansion Breakout
    # ─────────────────────────────────────────────────────
    avg_range  = (h - l).rolling(10).mean()
    was_tight  = (h.shift(1) - l.shift(1)) < avg_range * 0.5
    expanding  = (h - l) > avg_range * 2
    df["D14_Range_Expand"] = (
        was_tight & expanding & (c > o)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D15 — Retest of Broken Level
    # ─────────────────────────────────────────────────────
    old_res = h.rolling(20).max().shift(5)
    df["D15_Retest"] = (
        (c.shift(2) > old_res.shift(2)) &
        (abs(c - old_res) / c < 0.002) &
        (c > c.shift(1))
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D16 — Premium / Discount Entry
    # ─────────────────────────────────────────────────────
    day_high  = df.groupby("Date")["High"].transform("max")
    day_low   = df.groupby("Date")["Low"].transform("min")
    midpoint  = day_low + (day_high - day_low) * 0.5
    df["D16_Premium_Discount"] = (
        (c < midpoint) &
        (c > c.shift(1)) &
        (c > o) &
        (rsi < 45)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D17 — Volume Spike Continuation
    # ─────────────────────────────────────────────────────
    df["D17_Vol_Spike"] = (
        (vol > vma * 2.0) &
        (c > o) &
        (ema9 > ema21)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D18 — Failed Breakdown Long
    # ─────────────────────────────────────────────────────
    support = l.rolling(20).min().shift(1)
    df["D18_Failed_Breakdown"] = (
        (l.shift(1) < support) &
        (c > support) &
        (vol > vma) & (c > o)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D19 — Reversal at Previous Day Low
    # ─────────────────────────────────────────────────────
    df["D19_HTF_Reversal"] = (
        (abs(l - pdl) / pdl < 0.003) &
        (c > pdl) & (c > o) &
        (rsi > rsi.shift(1)) & (rsi < 45)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D20 — News Spike Rejection (bearish proxy)
    # ─────────────────────────────────────────────────────
    df["D20_News_Reject"] = (
        (h > h.shift(1) * 1.01) &
        (c < h * 0.995) &
        (uw > 0.5) &
        (c < o)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D21 — Clean Multi-Factor Confluence
    # ─────────────────────────────────────────────────────
    df["D21_Clean_Confluence"] = (
        (macd > msig) &
        (rsi > 45) & (rsi < 65) &
        (c > vwap) &
        (ema9 > ema21) &
        (vol > vma)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D22 — Breakout Confirmed Entry
    # ─────────────────────────────────────────────────────
    resistance = h.rolling(15).max().shift(1)
    df["D22_Breakout_Confirm"] = (
        (c > resistance) &
        (c > resistance * 1.002) &
        (vol > vma * 1.5)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D23 — Risk-Based Trend Follow
    # ─────────────────────────────────────────────────────
    atr_avg = atr.rolling(20).mean()
    df["D23_Risk_Trend"] = (
        (ema9 > ema21) &
        (c > ema9) &
        (atr < atr_avg * 1.5) &
        (c > vwap)
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D24 — Liquidity Sweep + Reversal
    # ─────────────────────────────────────────────────────
    df["D24_Liq_Sweep"] = (
        (l < pdl) &
        (c > pdl) &
        (c > o) &
        (c > o.shift(1)) &
        (o < c.shift(1))
    ).astype(int)

    # ─────────────────────────────────────────────────────
    # D25 — Kill Zone Setup (open 0-30 min / close 315+ min)
    # ─────────────────────────────────────────────────────
    in_kz   = (mso <= 30) | (mso >= 315)
    low5_kz = l.rolling(5).min().shift(1)
    df["D25_Kill_Zone"] = (
        in_kz &
        (l < low5_kz) &
        (c > low5_kz)
    ).astype(int)

    # ── Confluence Score (0-25) ───────────────────────────────
    avail = [s for s in INTRADAY_SIGNALS if s in df.columns]
    df["Total_Score"]             = df[avail].sum(axis=1)
    df["Intraday_Strong_Signal"]  = (df["Total_Score"] >= 5).astype(int)

    # ── Print frequency ──────────────────────────────────────
    total = len(df)
    print(f"\n{'Signal':<25} {'Fired':>7} {'% Days':>8}")
    print("─" * 42)
    for col in avail:
        n = df[col].sum()
        print(f"  {col:<23} {n:>7} {n/total*100:>7.1f}%")

    print(f"\n✅ Total intraday signals added: 25")
    print(f"   Max confluence today    : {df['Total_Score'].max()}")
    print(f"   Strong signal rows      : "
          f"{df['Intraday_Strong_Signal'].sum()}")

    return df

# EXPLANATION: D1 ORB, D3 VWAP pullback, D21 confluence
# — yeh teen NSE intraday trading ke core setups hain.
# D25 kill zone (pehle 30 min aur aakhri 1 ghanta)
# mein market mein sabse zyada movement hoti hai.