# =============================================================
# FILE: trading_system/phase_b/saver_b.py
# PURPOSE: Task 10 — Phase B outputs save karo
# =============================================================

import json
import joblib
import pandas as pd
from config_b import MODELS_DIR_B, DATA_PROC_B
from confluence import STRATEGY_COLS


def save_phase_b_outputs(best_model, scaler_b,
                         df: pd.DataFrame,
                         best_name: str, best_f1: float):
    """
    Phase B model, scaler, CSV aur strategy columns save karo.
    Ye sab Phase C (backtesting) mein use honge.

    Args:
        best_model: Best Phase B trained model.
        scaler_b  : Phase B StandardScaler.
        df        : Full featured DataFrame.
        best_name : Winning model name string.
        best_f1   : Winner's F1 score.
    """
    print("\n" + "─" * 55)
    print("💾 SAVING PHASE B OUTPUTS")
    print("─" * 55)

    saved = []

    try:
        # ── 10a: Best Phase B model ──
        model_path = MODELS_DIR_B / "best_model_phase_b.pkl"
        joblib.dump(best_model, model_path)
        saved.append(model_path)
        print(f"✅ Model   → {model_path.name}")

        # ── 10b: Phase B scaler ──
        scaler_path = MODELS_DIR_B / "scaler_phase_b.pkl"
        joblib.dump(scaler_b, scaler_path)
        saved.append(scaler_path)
        print(f"✅ Scaler  → {scaler_path.name}")

        # ── 10c: Full featured DataFrame ──
        csv_path = DATA_PROC_B / "RELIANCE_phase_b.csv"
        df.to_csv(csv_path, index=False)
        saved.append(csv_path)
        print(f"✅ CSV     → {csv_path.name}")

        # ── 10d: Strategy column names as JSON ──
        json_path = MODELS_DIR_B / "strategy_columns.json"
        with open(json_path, "w") as f:
            json.dump(STRATEGY_COLS, f, indent=2)
        saved.append(json_path)
        print(f"✅ JSON    → {json_path.name}")

    except Exception as e:
        print(f"❌ Save failed: {e}")
        raise

    # ── Final summary ──
    print("\n" + "═" * 55)
    print("✅ Phase B Complete! All outputs saved.")
    print("═" * 55)
    print(f"🏆 Best Model : {best_name}")
    print(f"   F1 Score   : {best_f1:.4f}")
    print("\nSaved files:")
    for f in saved:
        print(f"  📁 {f}")

# EXPLANATION: strategy_columns.json zaroori hai
# Phase C ke liye — backtester ko pata hona chahiye
# kaunsi columns strategy signals hain. Model aur
# scaler Phase D (real-time) mein load honge.