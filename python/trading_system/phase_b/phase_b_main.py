# =============================================================
# FILE: trading_system/phase_b/phase_b_main.py
# PURPOSE: Phase B main runner — sab modules order mein
#          Command: py -3.11 phase_b_main.py
# =============================================================

import warnings
warnings.filterwarnings("ignore")

from config_b      import CONFIG
from loader_b      import load_phase_a_data
from indicators_b  import add_new_indicators
from strategies    import add_strategy_signals
from confluence    import add_confluence_score
from visualizer_b  import plot_strategy_signals
from model_b       import (prepare_phase_b_features,
                           train_phase_b_models,
                           evaluate_and_compare)
from reporter      import generate_strategy_report
from saver_b       import save_phase_b_outputs


if __name__ == "__main__":
    print("=" * 55)
    print("  PHASE B — STRATEGY INTELLIGENCE LAYER")
    print("  Building on Phase A outputs...")
    print("=" * 55)

    # ── Task 1: Load Phase A data ──
    df, model_a, scaler_a = load_phase_a_data()

    # ── Task 2: New indicators ──
    df = add_new_indicators(df, CONFIG)

    # ── Task 3: 15 strategy signals ──
    df = add_strategy_signals(df, CONFIG)

    # ── Task 4: Confluence score ──
    df = add_confluence_score(df)

    # ── Task 5: Visualize signals ──
    plot_strategy_signals(df)

    # ── Task 6: Prepare Phase B features ──
    X_train, X_test, y_train, y_test, scaler_b, feat_cols = \
        prepare_phase_b_features(df)

    # ── Task 7: Train Phase B models ──
    rf_b, gb_b = train_phase_b_models(X_train, y_train)

    # ── Task 8: Evaluate + compare ──
    results, best_model, best_name = evaluate_and_compare(
        model_a, rf_b, gb_b,
        df, X_test, y_test,
        scaler_a, feat_cols
    )

    # ── Task 9: Strategy report ──
    generate_strategy_report(df)

    # ── Task 10: Save outputs ──
    best_f1 = max(results, key=lambda r: r["f1"])["f1"]
    save_phase_b_outputs(best_model, scaler_b, df, best_name, best_f1)