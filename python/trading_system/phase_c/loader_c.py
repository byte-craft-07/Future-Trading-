# =============================================================
# FILE: trading_system/phase_c/loader_c.py
# PURPOSE: Task 1 — Phase B CSV + model + scaler load karo
# =============================================================

import json
import joblib
import pandas as pd
from config_c import (PHASE_B_CSV, PHASE_B_MODEL,
                      PHASE_B_SCALER, PHASE_B_STRAT_J,
                      PHASE_B_STRATEGIES)


def load_phase_b_data():
    """
    Phase B ke saved outputs load karo.
    CSV, trained model, scaler, strategy columns JSON.

    Returns:
        df       : Phase B full featured DataFrame
        model_b  : Phase B best model
        scaler_b : Phase B StandardScaler
    """
    print("\n" + "─" * 58)
    print("📂 LOADING PHASE B OUTPUTS")
    print("─" * 58)

    # ── CSV load ──
    try:
        df = pd.read_csv(PHASE_B_CSV, parse_dates=["Date"])
        print(f"✅ CSV loaded     : {PHASE_B_CSV.name}")
        print(f"   Shape          : {df.shape}")
        print(f"   Date range     : "
              f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    except FileNotFoundError:
        print(f"❌ Phase B CSV not found: {PHASE_B_CSV}")
        print("   Phase B pehle successfully run karo!")
        raise

    # ── Model load ──
    try:
        model_b = joblib.load(PHASE_B_MODEL)
        print(f"✅ Model loaded   : {PHASE_B_MODEL.name}")
    except FileNotFoundError:
        print(f"❌ Model not found: {PHASE_B_MODEL}")
        raise

    # ── Scaler load ──
    try:
        scaler_b = joblib.load(PHASE_B_SCALER)
        print(f"✅ Scaler loaded  : {PHASE_B_SCALER.name}")
    except FileNotFoundError:
        print(f"❌ Scaler not found: {PHASE_B_SCALER}")
        raise

    # ── Strategy JSON load ──
    try:
        with open(PHASE_B_STRAT_J, "r") as f:
            strat_cols = json.load(f)
        print(f"✅ Strategies JSON: {len(strat_cols)} columns confirmed")
    except FileNotFoundError:
        print(f"⚠️  Strategy JSON not found — using default list")
        strat_cols = PHASE_B_STRATEGIES

    # ── Verify required columns ──
    required = ["Close", "Open", "High", "Low", "Volume",
                "RSI_14", "MACD", "SMA_20", "SMA_50", "EMA_20",
                "ATR_14", "Supertrend_Signal", "VWAP_Daily",
                "Confluence_Score", "Target"] + PHASE_B_STRATEGIES

    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"⚠️  Missing columns: {missing}")
    else:
        print("✅ All required columns verified.")

    # Print strategy columns found
    s_found = [c for c in PHASE_B_STRATEGIES if c in df.columns]
    print(f"\n   S1-S15 found   : {len(s_found)}/15")
    print("✅ Phase B data loaded successfully.\n")

    return df, model_b, scaler_b

# EXPLANATION: Phase C Phase B ke upar build hota
# hai. 15 strategy signals pehle se hain, ab 8
# structure signals aur add honge = 23 total.
# Backtesting prove karega ki in signals ne 2018-2024
# mein actually paise banaye ya nahi.