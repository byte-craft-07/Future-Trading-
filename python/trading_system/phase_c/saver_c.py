# =============================================================
# FILE: trading_system/phase_c/saver_c.py
# PURPOSE: Task 10 — Phase C outputs save karo
# =============================================================

import json
import joblib
import pandas as pd

from config_c import (MODELS_DIR_C, DATA_PROC_C,
                      BT_RESULTS_DIR, ALL_STRATEGIES,
                      PHASE_C_STRATEGIES)


def save_phase_c_outputs(model_c, scaler_c,
                         df: pd.DataFrame,
                         results: dict,
                         top_strategies: list):
    """
    Phase C model, scaler, CSV, backtest results save karo.
    Phase D real-time system inhe load karega.

    Args:
        model_c       : Phase C trained model.
        scaler_c      : Phase C StandardScaler.
        df            : Full featured DataFrame.
        results       : Backtest results dict.
        top_strategies: Top strategies for Phase D.
    """
    print("\n" + "─" * 58)
    print("💾 SAVING PHASE C OUTPUTS")
    print("─" * 58)

    saved = []

    try:
        # ── 10a: Model ──
        mp = MODELS_DIR_C / "best_model_phase_c.pkl"
        joblib.dump(model_c, mp)
        saved.append(mp)
        print(f"✅ Model      → {mp.name}")

        # ── 10b: Scaler ──
        sp = MODELS_DIR_C / "scaler_phase_c.pkl"
        joblib.dump(scaler_c, sp)
        saved.append(sp)
        print(f"✅ Scaler     → {sp.name}")

        # ── 10c: Featured DataFrame ──
        cp = DATA_PROC_C / "RELIANCE_phase_c.csv"
        df.to_csv(cp, index=False)
        saved.append(cp)
        print(f"✅ CSV        → {cp.name}")

        # ── 10d: All backtest results (stripped for JSON) ──
        results_clean = {}
        for name, m in results.items():
            results_clean[name] = {
                k: v for k, v in m.items()
                if k not in ["equity_curve", "dates", "trades"]
            }
        rp = BT_RESULTS_DIR / "all_strategy_results.json"
        with open(rp, "w") as f:
            json.dump(results_clean, f, indent=2)
        saved.append(rp)
        print(f"✅ BT Results → {rp.name}")

        # ── 10e: Top strategies for Phase D ──
        tp = BT_RESULTS_DIR / "top_strategies.json"
        with open(tp, "w") as f:
            json.dump(top_strategies, f, indent=2)
        saved.append(tp)
        print(f"✅ Top Strats → {tp.name}")

        # ── 10f: All signal column names ──
        all_sigs = [s for s in ALL_STRATEGIES if s in df.columns]
        ap = MODELS_DIR_C / "all_signal_columns.json"
        with open(ap, "w") as f:
            json.dump(all_sigs, f, indent=2)
        saved.append(ap)
        print(f"✅ All Signals → {ap.name}")

    except Exception as e:
        print(f"❌ Save failed: {e}")
        raise

    # ── Final message ──
    print("\n" + "═" * 58)
    print("✅ Phase C Complete! All outputs saved.")
    print("═" * 58)
    print(f"\n🚀 Top strategies for Phase D:")
    for i, s in enumerate(top_strategies, 1):
        print(f"   {i}. {s}")
    print("\nSaved files:")
    for f in saved:
        print(f"  📁 {f}")