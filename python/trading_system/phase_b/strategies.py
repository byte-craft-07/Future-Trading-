# =============================================================
# FILE: trading_system/phase_b/strategies.py
# PURPOSE: Task 3 — 15 trading strategies → binary columns
# RULE: shift(1) use karo prev candle data ke liye
#       (no look-ahead bias / data leakage)
# =============================================================

import pandas as pd


def add_strategy_signals(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    15 trading strategies ko binary signal columns mein
    convert karo. 1 = strategy BUY signal ON, 0 = OFF.

    Args:
        df    : DataFrame with all indicators.
        config: CONFIG dict with thresholds.

    Returns:
        DataFrame with S1 to S15 columns added.
    """
    print("\n" + "─" * 55)
    print("🧠 ADDING 15 STRATEGY SIGNALS")
    print("─" * 55)

    # ── Shorthand variables (makes code readable) ──
    c    = df["Close"]
    o    = df["Open"]
    h    = df["High"]
    l    = df["Low"]
    vol  = df["Volume"]
    rsi  = df["RSI_14"]
    macd = df["MACD"]
    msig = df["MACD_Signal"]
    mhst = df["MACD_Hist"]
    sma20= df["SMA_20"]
    sma50= df["SMA_50"]
    ema20= df["EMA_20"]
    vma  = df["Volume_MA_20"]
    bsize= df["Body_Size"]
    bu   = df["Bullish_Engulf"]
    st   = df["Supertrend_Signal"]
    vwap = df["VWAP_Daily"]

    ro   = config["rsi_oversold"]
    rob  = config["rsi_overbought"]
    rml  = config["rsi_midzone_low"]
    rmh  = config["rsi_midzone_high"]
    vsm  = config["volume_spike_mult"]
    sr   = config["sr_lookback"]

    # ─────────────────────────────────────────────────────────
    # S1: RSI + MACD Reversal
    # RSI oversold + MACD bullish cross + volume confirm
    # ─────────────────────────────────────────────────────────
    df["S1_RSI_MACD_Reversal"] = (
        (rsi < ro) &
        (macd > msig) &
        (vol > vma)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # S2: SMA Golden Cross Trend
    # SMA20 > SMA50 AND price above both MAs
    # ─────────────────────────────────────────────────────────
    df["S2_SMA_Crossover"] = (
        (sma20 > sma50) &
        (c > sma20) &
        (c > sma50)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # S3: RSI Pullback in Uptrend
    # Price in uptrend, RSI pulled back to 40-50 zone
    # ─────────────────────────────────────────────────────────
    df["S3_RSI_Pullback"] = (
        (c > sma50) &
        (rsi >= rml) &
        (rsi <= rmh)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # S4: MACD Histogram Growing Momentum
    # MACD bullish + histogram positive + growing
    # ─────────────────────────────────────────────────────────
    df["S4_MACD_Momentum"] = (
        (macd > msig) &
        (mhst > 0) &
        (mhst > mhst.shift(1))          # Histogram growing
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # S5: Volume Breakout at 20-Day High
    # Price breaks 20-day high + volume spike
    # shift(1) — prev day ka 20D max (no leakage)
    # ─────────────────────────────────────────────────────────
    rolling_20_high = c.rolling(20).max().shift(1)
    df["S5_Volume_Breakout"] = (
        (c > rolling_20_high) &
        (vol > vma * vsm)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # S6: Bullish Engulfing + MACD Confirm
    # Candle pattern + MACD agree + not overbought
    # ─────────────────────────────────────────────────────────
    df["S6_Engulf_MACD"] = (
        (bu == 1) &
        (macd > msig) &
        (rsi < rob)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # S7: Oversold Bounce Recovery
    # Previous day RSI < 30, today RSI recovered + volume up
    # shift(1) = prev day RSI (no leakage)
    # ─────────────────────────────────────────────────────────
    df["S7_Oversold_Bounce"] = (
        (rsi.shift(1) < ro) &           # Prev day oversold
        (rsi >= ro) &                   # Today recovered
        (vol > vma)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # S8: Clean Trend Follow
    # All trend indicators aligned bullish
    # ─────────────────────────────────────────────────────────
    df["S8_Trend_Follow"] = (
        (sma20 > sma50) &
        (c > sma20) &
        (macd > 0) &
        (macd > msig)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # S9: Momentum Breakout (10-Day)
    # 10-day high break + strong candle + volume
    # ─────────────────────────────────────────────────────────
    rolling_10_high  = c.rolling(10).max().shift(1)
    rolling_10_body  = bsize.rolling(10).mean()
    df["S9_Momentum_Breakout"] = (
        (c > rolling_10_high) &
        (bsize > rolling_10_body) &     # Strong candle body
        (vol > vma * 1.3)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # S10: EMA Crossover (Fast EMA rising above Slow SMA)
    # ─────────────────────────────────────────────────────────
    df["S10_EMA_Cross"] = (
        (ema20 > sma50) &
        (c > ema20) &
        (ema20 > ema20.shift(1))        # EMA rising
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # S11: Gap Up + Green Candle Hold
    # Opens above prev close by 0.5%, closes green + volume
    # ─────────────────────────────────────────────────────────
    gap_up = o > c.shift(1) * 1.005    # 0.5% gap up
    df["S11_Gap_Up"] = (
        gap_up &
        (c > o) &                       # Closed green
        (vol > vma)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # S12: Support Bounce Setup
    # Price near 20-day low (support) + oversold approach
    # shift(1) = prev day's 20D low (no leakage)
    # ─────────────────────────────────────────────────────────
    rolling_low   = l.rolling(sr).min().shift(1)
    near_support  = abs(c - rolling_low) / c < 0.02  # Within 2%
    df["S12_Support_Bounce"] = (
        near_support &
        (rsi < 40) &
        (c > o)                         # Bullish close
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # S13: EMA Pullback Entry in Uptrend
    # Price touched EMA20 but closed above it (support bounce)
    # ─────────────────────────────────────────────────────────
    df["S13_Pullback_Entry"] = (
        (c > sma50) &
        (l <= ema20) &                  # Wick touched EMA
        (c > ema20) &                   # But closed above
        (rsi > 40)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # S14: Tiny Candle Compression (Breakout Setup)
    # Very small candle (30% of prev) in uptrend = coiling
    # ─────────────────────────────────────────────────────────
    candle_range = (h - l) / c * 100
    prev_range   = candle_range.shift(1)
    tiny_candle  = candle_range < (prev_range * config["tiny_candle_pct"])
    df["S14_Tiny_Candle"] = (
        tiny_candle &
        (c > sma20)
    ).astype(int)

    # ─────────────────────────────────────────────────────────
    # S15: Supertrend + VWAP + Volume
    # All 3 aligned bullish = strong signal
    # ─────────────────────────────────────────────────────────
    df["S15_Supertrend"] = (
        (st == 1) &
        (c > vwap) &
        (vol > vma)
    ).astype(int)

    # ── Signal frequency report ───────────────────────────────
    strategy_cols = [c for c in df.columns if c.startswith("S") and "_" in c
                     and c[1:3].isdigit()]
    total = len(df)
    print(f"\n{'Strategy':<25} {'Signals':>8} {'% of Days':>10}")
    print("─" * 45)
    for col in sorted(strategy_cols):
        n_sig = df[col].sum()
        pct   = n_sig / total * 100
        print(f"{col:<25} {n_sig:>8} {pct:>9.1f}%")

    return df

# EXPLANATION: Har strategy = ek column. 1 matlab
# "aaj ye strategy ka BUY signal hai". Model in
# columns se seekhega — kab multiple strategies
# ek saath agree karti hain aur price actually
# upar jaati hai. shift(1) use kiya taaki prev
# candle ka data leak na ho (cheating nahi hogi).