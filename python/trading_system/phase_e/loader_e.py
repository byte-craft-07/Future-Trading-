# =============================================================
# FILE: trading_system/phase_e/loader_e.py
# PURPOSE: Task 1 — Load all previous phase outputs
# =============================================================

import json
import joblib
import pandas as pd
from config_e import (PHASE_D_CSV, PHASE_D_CLF, PHASE_D_REG,
                      PHASE_D_SCALER, PHASE_D_FEATURES,
                      PHASE_C_TOP_STRAT,
                      PHASE_B_SIGNALS, PHASE_C_SIGNALS,
                      PHASE_D_SIGNALS)


def load_all_phases():
    """
    Saari previous phases ke outputs ek jagah load karo.
    CSV, models, scalers, strategy JSON.

    Returns:
        df, ml_clf, ml_reg, scaler_d, feat_cols, top_strats
    """
    print("\n" + "─" * 58)
    print("📦 LOADING ALL PREVIOUS PHASE OUTPUTS")
    print("─" * 58)

    # ── Phase D CSV ──
    try:
        df = pd.read_csv(PHASE_D_CSV)
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"],
                                             utc=True,
                                             errors="coerce")
        print(f"✅ Phase D CSV    : {df.shape[0]} rows × "
              f"{df.shape[1]} cols")
    except FileNotFoundError:
        print(f"❌ Phase D CSV not found: {PHASE_D_CSV}")
        print("   Phase D --mode train pehle run karo!")
        raise

    # ── Phase D Models ──
    try:
        ml_clf    = joblib.load(PHASE_D_CLF)
        ml_reg    = joblib.load(PHASE_D_REG)
        scaler_d  = joblib.load(PHASE_D_SCALER)
        with open(PHASE_D_FEATURES) as f:
            feat_cols = json.load(f)
        print(f"✅ Phase D models : classifier + regressor loaded")
        print(f"   Feature cols   : {len(feat_cols)}")
    except FileNotFoundError as e:
        print(f"❌ Phase D model missing: {e}")
        raise

    # ── Phase C Top Strategies ──
    top_strats = []
    try:
        with open(PHASE_C_TOP_STRAT) as f:
            top_strats = json.load(f)
        print(f"✅ Phase C top    : {top_strats[:3]}")
    except FileNotFoundError:
        print("⚠️  Phase C top_strategies.json not found — skipping")

    # ── Verify signal columns ──
    def check_signals(col_list, label):
        found = [c for c in col_list if c in df.columns]
        print(f"✅ {label}: {len(found)}/{len(col_list)} loaded")
        return len(found)

    b_cnt = check_signals(PHASE_B_SIGNALS, "Phase B signals (S1-S15)")
    c_cnt = check_signals(PHASE_C_SIGNALS, "Phase C signals (C1-C8) ")
    d_cnt = check_signals(PHASE_D_SIGNALS, "Phase D signals (D1-D25)")

    total_so_far = b_cnt + c_cnt + d_cnt
    print(f"\n   Total signals so far : {total_so_far}")
    print(f"   Phase E will add     : 30 more")
    print(f"   Final total          : {total_so_far + 30}")

    return df, ml_clf, ml_reg, scaler_d, feat_cols, top_strats

# EXPLANATION: Pehle saari cheezein load karte hain.
# Phase D ne 5min intraday data + 25 D-signals banaye.
# Phase E unke UPAR 30 session signals add karega.
# Phir LSTM models train honge same data pe.