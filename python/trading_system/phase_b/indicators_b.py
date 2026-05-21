# =============================================================
# FILE: trading_system/phase_b/indicators_b.py
# PURPOSE: Task 2 — ATR, Supertrend, VWAP, Candle patterns
# NOTE: 'ta' library use kar rahe hain (pandas_ta ki jagah)
#       Supertrend manually implement kiya hai
# =============================================================

import pandas as pd
import numpy as np
import ta


def compute_supertrend(close: pd.Series, high: pd.Series,
                       low: pd.Series, atr: pd.Series,
                       multiplier: float = 3.0) -> pd.Series:
    """
    Supertrend indicator manually calculate karo.
    (ta library mein available nahi hai)

    Logic:
      Upper Band = (High+Low)/2 + multiplier * ATR
      Lower Band = (High+Low)/2 - multiplier * ATR
      1 = Bullish (price above lower band)
      0 = Bearish (price below upper band)

    Args:
        close, high, low: OHLC series
        atr             : Pre-calculated ATR series
        multiplier      : ATR multiplier (default 3.0)

    Returns:
        Series of 1 (bullish) / 0 (bearish)
    """
    # ── Numpy arrays for fast loop ──
    c   = close.values
    h   = high.values
    l   = low.values
    atr_arr = atr.values
    n   = len(c)

    hl2         = (h + l) / 2.0
    upper_basic = hl2 + multiplier * atr_arr
    lower_basic = hl2 - multiplier * atr_arr

    # Final bands (will be updated in loop)
    final_upper = upper_basic.copy()
    final_lower = lower_basic.copy()
    signal      = np.ones(n, dtype=int)   # Start bullish

    for i in range(1, n):
        # Skip NaN ATR rows
        if np.isnan(atr_arr[i]):
            signal[i] = 0
            continue

        # ── Final Upper Band ──
        if upper_basic[i] < final_upper[i-1] or c[i-1] > final_upper[i-1]:
            final_upper[i] = upper_basic[i]
        else:
            final_upper[i] = final_upper[i-1]

        # ── Final Lower Band ──
        if lower_basic[i] > final_lower[i-1] or c[i-1] < final_lower[i-1]:
            final_lower[i] = lower_basic[i]
        else:
            final_lower[i] = final_lower[i-1]

        # ── Direction decision ──
        if signal[i-1] == 1:        # Was bullish
            signal[i] = 1 if c[i] >= final_lower[i] else 0
        else:                        # Was bearish
            signal[i] = 1 if c[i] > final_upper[i] else 0

    return pd.Series(signal, index=close.index)


def add_new_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    4 naye indicators add karo jo Phase B strategies ko
    chahiye:
      - ATR_14          (volatility)
      - Supertrend_Signal (trend direction)
      - VWAP_Daily      (fair price approximation)
      - Candle_Body_Ratio (candle strength)
      - Bullish_Engulf  (candle pattern)

    Args:
        df    : Phase A DataFrame
        config: CONFIG dict

    Returns:
        DataFrame with new indicator columns.
    """
    print("\n" + "─" * 55)
    print("📐 ADDING NEW INDICATORS (Phase B)")
    print("─" * 55)

    # ── 2a: ATR (Average True Range) ─────────────────────────
    # Measures daily price movement range
    atr_obj      = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"],
        window=config["atr_period"]
    )
    df["ATR_14"] = atr_obj.average_true_range()

    # ── 2b: Supertrend ───────────────────────────────────────
    # Green (1) = bullish trend, Red (0) = bearish trend
    df["Supertrend_Signal"] = compute_supertrend(
        df["Close"], df["High"], df["Low"],
        df["ATR_14"],
        multiplier=config["supertrend_mult"]
    )

    # ── 2c: Daily VWAP Approximation ─────────────────────────
    # True VWAP needs tick data (Phase D mein hoga)
    # Approximation: typical price = (H+L+C)/3
    df["VWAP_Daily"] = (df["High"] + df["Low"] + df["Close"]) / 3

    # ── 2d: Candle Body Ratio ─────────────────────────────────
    # Near 1 = strong full-body candle
    # Near 0 = doji / indecision candle
    df["Candle_Body_Ratio"] = (
        abs(df["Close"] - df["Open"]) /
        (df["High"] - df["Low"] + 0.0001)
    )

    # ── 2e: Bullish Engulfing Pattern ─────────────────────────
    # Previous candle bearish AND current candle bullish
    # AND current fully engulfs previous
    prev_open  = df["Open"].shift(1)
    prev_close = df["Close"].shift(1)

    df["Bullish_Engulf"] = (
        (prev_close < prev_open) &          # Prev bearish
        (df["Close"] > df["Open"]) &        # Curr bullish
        (df["Open"]  < prev_close) &        # Gap down open
        (df["Close"] > prev_open)           # Close above prev open
    ).astype(int)

    # ── Drop NaN rows ─────────────────────────────────────────
    rows_before = len(df)
    df = df.dropna(subset=["ATR_14"]).reset_index(drop=True)

    new_cols = ["ATR_14", "Supertrend_Signal", "VWAP_Daily",
                "Candle_Body_Ratio", "Bullish_Engulf"]
    print(f"New columns added  : {new_cols}")
    print(f"Rows dropped (NaN) : {rows_before - len(df)}")
    print(f"Final shape        : {df.shape}")

    return df

# EXPLANATION: ATR = daily average volatility.
# Supertrend manually implement kiya — ta library
# mein nahi hai. VWAP = intraday "fair price" ka
# daily approximation. Bullish Engulf = ek candle
# pattern jo reversal signal deta hai.