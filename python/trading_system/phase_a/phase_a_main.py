# =============================================================
# FILE: trading_system/phase_a/phase_a_main.py
# PURPOSE: Main runner — sab modules ko order mein chalao
#          Command: python phase_a_main.py
# =============================================================

import warnings
warnings.filterwarnings("ignore")

# ── Apne modules import karo ─────────────────────────────────
from config      import CONFIG
from data_loader import load_or_download
from cleaner     import clean_data
from features    import add_features
from target      import create_target
from model_utils import (split_and_scale, train_models,
                         evaluate_model, compare_models)
from visualizer  import plot_price_chart
from saver       import save_outputs


if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE A — INTELLIGENT TRADING SYSTEM FOUNDATION")
    print(f"  Stock  : {CONFIG['ticker']}")
    print(f"  Period : {CONFIG['start_date']} → {CONFIG['end_date']}")
    print("=" * 60)

    # ── Step 1: Data Load/Download ──
    df_raw = load_or_download(CONFIG)

    # ── Step 2: Clean ──
    df_clean = clean_data(df_raw)

    # ── Step 3: Features ──
    df_features = add_features(df_clean)

    # ── Step 4: Target Labels ──
    df_labeled = create_target(df_features)

    # ── Step 5: Price Chart ──
    plot_price_chart(df_labeled, CONFIG)

    # ── Step 6: Split + Scale ──
    X_train, X_test, y_train, y_test, scaler, feature_cols = \
        split_and_scale(df_labeled, CONFIG)

    # ── Step 7: Train ──
    lr_model, rf_model = train_models(X_train, y_train)

    # ── Step 8: Evaluate ──
    results = []
    results.append(evaluate_model(
        lr_model, X_test, y_test,
        name="Logistic Regression"
    ))
    results.append(evaluate_model(
        rf_model, X_test, y_test,
        name="Random Forest",
        feature_cols=feature_cols
    ))
    compare_models(results)

    # ── Best model pick karo ──
    best_result = max(results, key=lambda r: r["f1"])
    best_model  = (rf_model if "Random" in best_result["model"]
                   else lr_model)

    # ── Step 9: Save ──
    save_outputs(best_model, scaler, df_labeled, CONFIG)