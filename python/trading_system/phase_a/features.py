# =============================================================
# FILE: trading_system/phase_a/features.py
# PURPOSE: Task 4 — Technical indicators add karo (ML features)
# NOTE: 'ta' library use kar rahe hain — pandas_ta ka
#       koi bhi version Python 3.11/3.14 pe install nahi hota
# =============================================================

import pandas as pd
import numpy as np
import ta              # pip install ta


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    'ta' library se technical indicators add karo.
    Ye sab columns ML model ka input (X) banenge.

    Args:
        df: Cleaned OHLCV DataFrame.

    Returns:
        DataFrame with all indicator columns added.
    """
    print("\n" + "─" * 50)
    print("⚙️  FEATURE ENGINEERING")
    print("─" * 50)

    # ── Moving Averages ──────────────────────────────────────
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)

    # ── Momentum: RSI ────────────────────────────────────────
    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)

    # ── Trend: MACD ──────────────────────────────────────────
    macd_obj          = ta.trend.MACD(df["Close"],
                                      window_fast=12,
                                      window_slow=26,
                                      window_sign=9)
    df["MACD"]        = macd_obj.macd()
    df["MACD_Signal"] = macd_obj.macd_signal()
    df["MACD_Hist"]   = macd_obj.macd_diff()

    # ── Volatility: Bollinger Bands ──────────────────────────
    bb_obj         = ta.volatility.BollingerBands(df["Close"],
                                                  window=20,
                                                  window_dev=2)
    df["BB_Upper"] = bb_obj.bollinger_hband()
    df["BB_Mid"]   = bb_obj.bollinger_mavg()
    df["BB_Lower"] = bb_obj.bollinger_lband()

    # ── Volume ───────────────────────────────────────────────
    df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()

    # ── Custom Features ──────────────────────────────────────

    # Price SMA_20 se kitna % door hai
    df["Price_vs_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"] * 100

    # Daily High-Low range — volatility (%)
    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"] * 100

    # Candle body size — momentum strength (%)
    df["Body_Size"] = abs(df["Close"] - df["Open"]) / df["Close"] * 100

    # ── Warm-up NaN rows drop karo ───────────────────────────
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)

    print(f"Rows dropped (NaN warm-up): {rows_before - len(df)}")
    print(f"Final shape               : {df.shape}")

    # ── Features list print ──────────────────────────────────
    raw_cols     = ["Date", "Open", "High", "Low", "Close",
                    "Adj Close", "Volume"]
    feature_cols = [c for c in df.columns if c not in raw_cols]
    print("\nFeatures added:")
    for col in feature_cols:
        print(f"  ✓ {col}")

    return df