# =============================================================
# FILE: trading_system/phase_d/saver_d.py
# PURPOSE: Task 9 — Phase D outputs save karo
# =============================================================

import json
import joblib
import pandas as pd
from config_d import MODELS_DIR_D, DATA_PROC_D, SIGNALS_DIR


def save_phase_d_outputs(clf, reg, scaler,
                          df: pd.DataFrame,
                          signal: dict,
                          feature_cols: list):
    """
    Classifier, regressor, scaler, CSV, signal JSON save karo.

    Args:
        clf, reg     : Trained models.
        scaler       : Fitted StandardScaler.
        df           : Full intraday featured DataFrame.
        signal       : Latest signal dict.
        feature_cols : Feature column name list.
    """
    print("\n" + "─" * 55)
    print("💾 SAVING PHASE D OUTPUTS")
    print("─" * 55)

    saved = []
    try:
        # ── 9a: Classifier ──
        p1 = MODELS_DIR_D / "classifier_phase_d.pkl"
        joblib.dump(clf, p1)
        saved.append(p1)
        print(f"✅ Classifier → {p1.name}")

        # ── 9b: Regressor ──
        p2 = MODELS_DIR_D / "regressor_phase_d.pkl"
        joblib.dump(reg, p2)
        saved.append(p2)
        print(f"✅ Regressor  → {p2.name}")

        # ── 9c: Scaler ──
        p3 = MODELS_DIR_D / "scaler_phase_d.pkl"
        joblib.dump(scaler, p3)
        saved.append(p3)
        print(f"✅ Scaler     → {p3.name}")

        # ── 9d: Featured intraday CSV ──
        p4 = DATA_PROC_D / "RELIANCE_5min_features.csv"
        df.to_csv(p4, index=False)
        saved.append(p4)
        print(f"✅ CSV        → {p4.name}")

        # ── 9e: Latest signal JSON ──
        p5 = SIGNALS_DIR / "latest_signal.json"
        with open(p5, "w") as f:
            json.dump(signal, f, indent=2, default=str)
        saved.append(p5)
        print(f"✅ Signal     → {p5.name}")

        # ── 9f: Feature columns JSON ──
        p6 = MODELS_DIR_D / "phase_d_features.json"
        with open(p6, "w") as f:
            json.dump(feature_cols, f, indent=2)
        saved.append(p6)
        print(f"✅ Features   → {p6.name}")

    except Exception as e:
        print(f"❌ Save failed: {e}")
        raise

    print("\n" + "═" * 55)
    print("✅ Phase D Complete!")
    print("🔮 10-min Predictor : Ready")
    print("📡 Live System      : Ready (--mode live)")
    print("🎯 Signal Engine    : Ready (--mode signal)")
    print("═" * 55)
    print("\nSaved files:")
    for f in saved:
        print(f"  📁 {f}")