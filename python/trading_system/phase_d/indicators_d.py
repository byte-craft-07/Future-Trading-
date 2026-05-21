# =============================================================
# FILE: trading_system/phase_d/indicators_d.py
# PURPOSE: Task 2 — VWAP (daily reset), ORB, PDH/PDL,
#          RSI, MACD, EMA, BB, ATR, Supertrend
# =============================================================

import numpy as np
import pandas as pd
import ta
from config_d import CONFIG


# ─────────────────────────────────────────────────────────────
# Manual Supertrend (ta library mein nahi hai)
# ─────────────────────────────────────────────────────────────
def compute_supertrend(close: pd.Series, high: pd.Series,
                       low: pd.Series, atr: pd.Series,
                       multiplier: float = 3.0) -> pd.Series:
    """Supertrend indicator — 1=bullish, 0=bearish."""
    c, h, l, a = close.values, high.values, low.values, atr.values
    n           = len(c)
    hl2         = (h + l) / 2.0
    ub          = hl2 + multiplier * a
    lb          = hl2 - multiplier * a
    final_ub    = ub.copy()
    final_lb    = lb.copy()
    signal      = np.ones(n, dtype=int)

    for i in range(1, n):
        if np.isnan(a[i]):
            signal[i] = 0
            continue
        final_ub[i] = (ub[i] if ub[i] < final_ub[i-1]
                       or c[i-1] > final_ub[i-1]
                       else final_ub[i-1])
        final_lb[i] = (lb[i] if lb[i] > final_lb[i-1]
                       or c[i-1] < final_lb[i-1]
                       else final_lb[i-1])
        if signal[i-1] == 1:
            signal[i] = 1 if c[i] >= final_lb[i] else 0
        else:
            signal[i] = 1 if c[i] > final_ub[i] else 0

    return pd.Series(signal, index=close.index)


# ─────────────────────────────────────────────────────────────
# VWAP — resets every day at 09:15
# ─────────────────────────────────────────────────────────────
def compute_daily_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Real intraday VWAP — resets at 09:15 every day.
    Formula: cumsum(TP * Volume) / cumsum(Volume)
    Typical Price (TP) = (H + L + C) / 3
    """
    vwap_parts = []
    for date, grp in df.groupby("Date"):
        tp           = (grp["High"] + grp["Low"] + grp["Close"]) / 3
        cum_tp_vol   = (tp * grp["Volume"]).cumsum()
        cum_vol      = grp["Volume"].cumsum()
        vwap         = cum_tp_vol / cum_vol.replace(0, np.nan)
        vwap_parts.append(vwap)
    return pd.concat(vwap_parts).reindex(df.index)


# ─────────────────────────────────────────────────────────────
# ORB — first 30 minutes of each day
# ─────────────────────────────────────────────────────────────
def compute_orb(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Opening Range Breakout levels.
    ORB_High = max(High) in first 30 min
    ORB_Low  = min(Low)  in first 30 min
    Applied to all candles of that day.
    """
    orb_h, orb_l = {}, {}
    orb_min = config["orb_minutes"]

    for date, grp in df.groupby("Date"):
        window = grp[grp["Minutes_Since_Open"] < orb_min]
        if len(window) > 0:
            orb_h[date] = window["High"].max()
            orb_l[date] = window["Low"].min()
        else:
            orb_h[date] = grp["High"].iloc[0]
            orb_l[date] = grp["Low"].iloc[0]

    df["ORB_High"] = df["Date"].map(orb_h)
    df["ORB_Low"]  = df["Date"].map(orb_l)
    return df


# ─────────────────────────────────────────────────────────────
# Previous Day Levels (PDH / PDL / PDC)
# ─────────────────────────────────────────────────────────────
def compute_previous_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    PDH = previous day High
    PDL = previous day Low
    PDC = previous day Close (last candle)
    """
    daily = df.groupby("Date").agg(
        PDH_raw=("High",  "max"),
        PDL_raw=("Low",   "min"),
        PDC_raw=("Close", "last")
    )
    # Shift 1 day to get PREVIOUS day levels
    daily_shifted = daily.shift(1)

    df["PDH"] = df["Date"].map(daily_shifted["PDH_raw"])
    df["PDL"] = df["Date"].map(daily_shifted["PDL_raw"])
    df["PDC"] = df["Date"].map(daily_shifted["PDC_raw"])
    return df


# ─────────────────────────────────────────────────────────────
# Main: Add all indicators
# ─────────────────────────────────────────────────────────────
def add_intraday_indicators(df: pd.DataFrame,
                             config: dict) -> pd.DataFrame:
    """
    All intraday indicators add karo:
    RSI, MACD, EMA, BB, ATR, Supertrend,
    VWAP (daily reset), ORB, PDH/PDL/PDC,
    Custom candle features.

    Args:
        df    : 5min OHLCV DataFrame.
        config: CONFIG dict.

    Returns:
        Enriched DataFrame.
    """
    print("\n" + "─" * 55)
    print("📐 ADDING INTRADAY INDICATORS")
    print("─" * 55)

    # ── 2a: Standard indicators ────────────────────────────
    df["RSI_14"]  = ta.momentum.rsi(df["Close"],
                                     window=config["rsi_period"])

    macd_obj      = ta.trend.MACD(df["Close"],
                                   window_fast   = config["macd_fast"],
                                   window_slow   = config["macd_slow"],
                                   window_sign   = config["macd_signal"])
    df["MACD"]        = macd_obj.macd()
    df["MACD_Signal"] = macd_obj.macd_signal()
    df["MACD_Hist"]   = macd_obj.macd_diff()

    df["EMA_9"]   = ta.trend.ema_indicator(df["Close"],
                                            window=config["ema_fast"])
    df["EMA_21"]  = ta.trend.ema_indicator(df["Close"],
                                            window=config["ema_slow"])

    bb            = ta.volatility.BollingerBands(df["Close"],
                                                  window    = config["bb_period"],
                                                  window_dev= 2)
    df["BB_Upper"]= bb.bollinger_hband()
    df["BB_Mid"]  = bb.bollinger_mavg()
    df["BB_Lower"]= bb.bollinger_lband()

    atr_obj       = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"],
        window=config["atr_period"]
    )
    df["ATR_14"]  = atr_obj.average_true_range()

    df["Supertrend_Signal"] = compute_supertrend(
        df["Close"], df["High"], df["Low"], df["ATR_14"],
        multiplier=config["supertrend_mult"]
    )

    df["Volume_MA_20"] = df["Volume"].rolling(20).mean()

    # ── 2b: Real VWAP (daily reset) ─────────────────────────
    df["VWAP"]     = compute_daily_vwap(df)

    # ── 2c: VWAP Deviation % ────────────────────────────────
    df["VWAP_Dev"] = (df["Close"] - df["VWAP"]) / df["VWAP"] * 100

    # ── 2d: Opening Range ────────────────────────────────────
    df = compute_orb(df, config)

    # ── 2e: Previous Day Levels ──────────────────────────────
    df = compute_previous_day_levels(df)

    # ── 2f: Custom candle features ───────────────────────────
    df["Body_Size"]   = (abs(df["Close"] - df["Open"])
                          / df["Close"] * 100)
    df["Upper_Wick"]  = ((df["High"] - df[["Open","Close"]].max(axis=1))
                          / df["Close"] * 100)
    df["Lower_Wick"]  = ((df[["Open","Close"]].min(axis=1) - df["Low"])
                          / df["Close"] * 100)
    df["Candle_Type"] = (df["Close"] > df["Open"]).astype(int)

    # ── 2g: Drop NaN rows ────────────────────────────────────
    rows_before = len(df)
    df = df.dropna(subset=["RSI_14", "MACD", "ATR_14",
                            "VWAP", "ORB_High"]).reset_index(drop=True)

    print(f"Rows dropped (NaN): {rows_before - len(df)}")
    print(f"Final shape       : {df.shape}")
    print(f"Indicators added  : RSI, MACD, EMA9/21, BB, ATR, "
          f"Supertrend, VWAP, ORB, PDH/PDL/PDC")

    return df

# EXPLANATION: VWAP har din 09:15 pe reset hoti hai.
# Yeh real intraday fair price hai. ORB (Opening Range
# Breakout) pehle 30 min ka high/low hai — iski
# breaking bahut strong signal hoti hai. PDH/PDL
# pichle din ke levels hain jo support/resistance ban
# sakte hain.