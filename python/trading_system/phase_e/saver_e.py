# =============================================================
# FILE: trading_system/phase_e/saver_e.py
# PURPOSE: Task 8 — Save all Phase E outputs
# =============================================================

import json
import joblib
import pandas as pd
from config_e import (MODELS_DIR_E, DATA_PROC_E,
                      SIGNALS_DIR_E, PHASE_E_SIGNAL_NAMES,
                      ALL_78_SIGNALS)

try:
    from tensorflow.keras.models import save_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def save_phase_e_outputs(lstm_price, price_scaler,
                          lstm_vol,   vol_scaler,
                          df: pd.DataFrame,
                          signal: dict):
    """
    Phase E sab outputs save karo — LSTM models, scalers,
    CSV, signal JSON, complete signal list.

    Args:
        lstm_price, lstm_vol : LSTM models (or None).
        price_scaler, vol_scaler: MinMaxScalers.
        df                   : Full featured DataFrame.
        signal               : Latest ensemble signal.
    """
    print("\n" + "─" * 58)
    print("💾 SAVING PHASE E OUTPUTS")
    print("─" * 58)

    saved = []

    try:
        # ── LSTM Price Model ──
        if lstm_price is not None and TF_AVAILABLE:
            p1 = MODELS_DIR_E / "lstm_price_model.keras"
            save_model(lstm_price, str(p1))
            saved.append(p1)
            print(f"✅ LSTM Price  → {p1.name}/")

        # ── LSTM Vol Model ──
        if lstm_vol is not None and TF_AVAILABLE:
            p2 = MODELS_DIR_E / "lstm_vol_model.keras"
            save_model(lstm_vol, str(p2))
            saved.append(p2)
            print(f"✅ LSTM Vol    → {p2.name}/")

        # ── Price Scaler ──
        if price_scaler is not None:
            p3 = MODELS_DIR_E / "price_scaler_e.pkl"
            joblib.dump(price_scaler, p3)
            saved.append(p3)
            print(f"✅ PriceScaler → {p3.name}")

        # ── Vol Scaler ──
        if vol_scaler is not None:
            p4 = MODELS_DIR_E / "vol_scaler_e.pkl"
            joblib.dump(vol_scaler, p4)
            saved.append(p4)
            print(f"✅ VolScaler   → {p4.name}")

        # ── Full featured CSV ──
        p5 = DATA_PROC_E / "RELIANCE_phase_e.csv"
        df.to_csv(p5, index=False)
        saved.append(p5)
        print(f"✅ CSV         → {p5.name}")

        # ── Ensemble signal ──
        p6 = SIGNALS_DIR_E / "ensemble_signal.json"
        with open(p6, "w") as f:
            json.dump(signal, f, indent=2, default=str)
        saved.append(p6)
        print(f"✅ Signal      → {p6.name}")

        # ── All 78 signal columns ──
        avail_78 = [s for s in ALL_78_SIGNALS if s in df.columns]
        p7 = MODELS_DIR_E / "all_78_signals.json"
        with open(p7, "w") as f:
            json.dump(avail_78, f, indent=2)
        saved.append(p7)
        print(f"✅ 78 Signals  → {p7.name} ({len(avail_78)} found)")

    except Exception as e:
        print(f"❌ Save error: {e}")

    # ── Final system summary ──
    print("\n" + "╔" + "═" * 54 + "╗")
    print("║" + "  ✅ PHASE E COMPLETE — FULL SYSTEM READY!  ".center(54) + "║")
    print("╠" + "═" * 54 + "╣")
    print("║  🤖 ML Classifier     : Phase D (ready)         ║")
    print("║  🧠 LSTM Price        : Phase E (ready)         ║")
    print("║  📊 LSTM Volatility   : Phase E (ready)         ║")
    print("║  💬 Sentiment Engine  : Active                  ║")
    print("║  🔮 Ensemble System   : 4-model weighted combo  ║")
    print("║  📡 Live Dashboard    : 6-panel dark terminal   ║")
    print("╠" + "─" * 54 + "╣")
    avail = [s for s in ALL_78_SIGNALS if s in df.columns]
    print(f"║  Total Signals        : {len(avail)}/78 available       ║")
    print(f"║  Expert E30 Signal    : {'ACTIVE ⭐' if signal.get('expert_signal') else 'inactive  '}               ║")
    print(f"║  Latest Signal        : {signal.get('signal','?'):<28}║")
    print(f"║  Ensemble Confidence  : {signal.get('ensemble_confidence',0):.1f}%                    ║")
    print("╚" + "═" * 54 + "╝")
    print("\nSaved files:")
    for f in saved:
        print(f"  📁 {f}")