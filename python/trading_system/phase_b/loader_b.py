# =============================================================
# FILE: trading_system/phase_b/loader_b.py
# PURPOSE: Task 1 — Phase A CSV + model + scaler load karo
# =============================================================

import pandas as pd
import joblib
from config_b import PHASE_A_CSV, PHASE_A_MODEL, PHASE_A_SCALER, PHASE_A_FEATURES


def load_phase_a_data():
    """
    Phase A ke saved outputs load karo:
      - Processed CSV (features + target)
      - Trained model
      - Fitted scaler

    Returns:
        df      : Phase A featured DataFrame
        model_a : Phase A best model (Random Forest)
        scaler_a: Phase A StandardScaler
    """
    print("\n" + "─" * 55)
    print("📂 LOADING PHASE A OUTPUTS")
    print("─" * 55)

    # ── CSV load karo ──
    try:
        df = pd.read_csv(PHASE_A_CSV, parse_dates=["Date"])
        print(f"✅ CSV loaded       : {PHASE_A_CSV.name}")
        print(f"   Shape            : {df.shape}")
        print(f"   Columns          : {list(df.columns)}")
    except FileNotFoundError:
        print(f"❌ CSV not found: {PHASE_A_CSV}")
        print("   Phase A pehle run karo!")
        raise

    # ── Phase A model load karo ──
    try:
        model_a = joblib.load(PHASE_A_MODEL)
        print(f"✅ Model A loaded   : {PHASE_A_MODEL.name}")
    except FileNotFoundError:
        print(f"❌ Model not found: {PHASE_A_MODEL}")
        raise

    # ── Phase A scaler load karo ──
    try:
        scaler_a = joblib.load(PHASE_A_SCALER)
        print(f"✅ Scaler A loaded  : {PHASE_A_SCALER.name}")
    except FileNotFoundError:
        print(f"❌ Scaler not found: {PHASE_A_SCALER}")
        raise

    # ── Required columns verify karo ──
    required = ["Close", "Open", "High", "Low", "Volume",
                "Target"] + PHASE_A_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"⚠️  Missing columns: {missing}")
    else:
        print(f"✅ All required columns present.")

    return df, model_a, scaler_a

# EXPLANATION: Phase A ka kaam waste nahi hoga.
# Hum uske features ke UPAR naye strategy features
# add karenge. Scaler aur model dono load karte hain
# taaki Phase A vs Phase B comparison ho sake.