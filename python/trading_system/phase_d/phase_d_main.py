# =============================================================
# FILE: trading_system/phase_d/phase_d_main.py
# PURPOSE: Phase D main runner with 3 modes
#
# USAGE:
#   py -3.11 phase_d_main.py --mode train    ← Data + models
#   py -3.11 phase_d_main.py --mode signal   ← One signal
#   py -3.11 phase_d_main.py --mode live     ← Live chart
# =============================================================

import argparse
import warnings
import joblib
import json
warnings.filterwarnings("ignore")

from config_d         import CONFIG, MODELS_DIR_D
from data_pipeline    import load_or_download_intraday
from indicators_d     import add_intraday_indicators
from signals_d        import add_intraday_signals
from predictor        import build_prediction_models
from signal_engine    import generate_signal, print_signal_box
from dashboard        import build_live_dashboard
from intraday_backtest import quick_backtest_intraday
from saver_d          import save_phase_d_outputs
from live_system      import run_live_system, fetch_and_process


def mode_train():
    """Full pipeline: data → features → signals → train → save."""
    print("=" * 55)
    print("  PHASE D — INTRADAY ML SYSTEM  [TRAIN MODE]")
    print("=" * 55)

    # Task 1: Data
    df = load_or_download_intraday(CONFIG)

    # Task 2: Indicators
    df = add_intraday_indicators(df, CONFIG)

    # Task 3: Signals
    df = add_intraday_signals(df, CONFIG)

    # Task 4: Build models
    clf, reg, scaler, feat_cols = build_prediction_models(df, CONFIG)

    # Task 5: Latest signal
    signal = generate_signal(df, clf, reg, scaler, feat_cols, CONFIG)
    print_signal_box(signal)

    # Task 6: Dashboard snapshot
    build_live_dashboard(df, signal, CONFIG, save=True)

    # Task 8: Quick backtest
    quick_backtest_intraday(df, CONFIG)

    # Task 9: Save
    save_phase_d_outputs(clf, reg, scaler, df, signal, feat_cols)


def mode_signal():
    """Load saved models, fetch latest data, print one signal."""
    print("=" * 55)
    print("  PHASE D — SIGNAL MODE")
    print("=" * 55)

    try:
        clf    = joblib.load(MODELS_DIR_D / "classifier_phase_d.pkl")
        reg    = joblib.load(MODELS_DIR_D / "regressor_phase_d.pkl")
        scaler = joblib.load(MODELS_DIR_D / "scaler_phase_d.pkl")
        with open(MODELS_DIR_D / "phase_d_features.json") as f:
            feat_cols = json.load(f)
        print("✅ Models loaded.")
    except FileNotFoundError:
        print("❌ Models not found. Run --mode train first!")
        return

    print("📡 Fetching latest 5min data...")
    df = fetch_and_process(CONFIG)
    if df is None:
        print("❌ Could not fetch data.")
        return

    signal = generate_signal(df, clf, reg, scaler, feat_cols, CONFIG)
    print_signal_box(signal)
    build_live_dashboard(df, signal, CONFIG, save=True)


def mode_live():
    """Load saved models, start live auto-refreshing system."""
    print("=" * 55)
    print("  PHASE D — LIVE SYSTEM MODE")
    print("=" * 55)

    try:
        clf    = joblib.load(MODELS_DIR_D / "classifier_phase_d.pkl")
        reg    = joblib.load(MODELS_DIR_D / "regressor_phase_d.pkl")
        scaler = joblib.load(MODELS_DIR_D / "scaler_phase_d.pkl")
        with open(MODELS_DIR_D / "phase_d_features.json") as f:
            feat_cols = json.load(f)
        print("✅ Models loaded. Starting live system...")
    except FileNotFoundError:
        print("❌ Models not found. Run --mode train first!")
        return

    run_live_system(clf, reg, scaler, feat_cols, CONFIG)


# ── Argument Parser ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase D — Intraday Trading System"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "signal", "live"],
        default="train",
        help="train=full pipeline | signal=one signal | live=live chart"
    )
    args = parser.parse_args()

    if args.mode == "train":
        mode_train()
    elif args.mode == "signal":
        mode_signal()
    elif args.mode == "live":
        mode_live()