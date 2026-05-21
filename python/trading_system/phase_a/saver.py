# =============================================================
# FILE: trading_system/phase_a/saver.py
# PURPOSE: Task 9 — Model, scaler, CSV disk pe save karo
# =============================================================

import joblib          # Sklearn model save/load
import pandas as pd    # CSV save

from config import MODELS_DIR, DATA_PROC


def save_outputs(best_model, scaler, df: pd.DataFrame, config: dict):
    """
    Best model, scaler aur featured DataFrame save karo.
    Ye files Phase B/C/D mein load hongi.

    Args:
        best_model: Winning sklearn model object.
        scaler    : Fitted StandardScaler.
        df        : Final labeled + featured DataFrame.
        config    : CONFIG dict (ticker name ke liye).
    """
    print("\n" + "─" * 50)
    print("💾 SAVING OUTPUTS")
    print("─" * 50)

    saved = []

    try:
        # ── Model save ──
        model_path = MODELS_DIR / "best_model_phase_a.pkl"
        joblib.dump(best_model, model_path)
        saved.append(model_path)
        print(f"✅ Model  → {model_path}")

        # ── Scaler save ──
        scaler_path = MODELS_DIR / "scaler_phase_a.pkl"
        joblib.dump(scaler, scaler_path)
        saved.append(scaler_path)
        print(f"✅ Scaler → {scaler_path}")

        # ── Featured CSV save ──
        ticker_clean = config["ticker"].replace(".", "_")
        csv_path     = DATA_PROC / f"{ticker_clean}_features.csv"
        df.to_csv(csv_path, index=False)
        saved.append(csv_path)
        print(f"✅ CSV    → {csv_path}")

    except Exception as e:
        print(f"❌ Save failed: {e}")
        raise

    # ── Success message ──
    print("\n" + "═" * 50)
    print("✅ Phase A Complete! All outputs saved.")
    print("═" * 50)
    for f in saved:
        print(f"  📁 {f}")