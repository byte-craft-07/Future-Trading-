# =============================================================
# FILE: trading_system/phase_e/phase_e_main.py
# PURPOSE: Phase E final system runner
#
# MODES:
#   py -3.11 phase_e_main.py --mode train   ← Full pipeline
#   py -3.11 phase_e_main.py --mode signal  ← One signal
#   py -3.11 phase_e_main.py --mode live    ← Live dashboard
# =============================================================

import argparse
import json
import warnings
import joblib
warnings.filterwarnings("ignore")

from config_e        import CONFIG, MODELS_DIR_E
from loader_e        import load_all_phases
from session_signals import add_session_signals
from lstm_models     import (build_lstm_price_model,
                              build_lstm_volatility_model,
                              lstm_predict_latest)
from ensemble_engine import (run_ensemble_prediction,
                              print_ensemble_signal)
from final_dashboard import build_final_dashboard
from saver_e         import save_phase_e_outputs


def mode_train():
    """Full Phase E pipeline."""
    print("=" * 58)
    print("  PHASE E — FINAL SYSTEM  [TRAIN MODE]")
    print("  LSTM + Session Signals + Ensemble")
    print("=" * 58)

    # Task 1: Load all phases
    df, ml_clf, ml_reg, scaler_d, feat_d, top_strats = \
        load_all_phases()

    # Task 2: 30 Session signals
    df = add_session_signals(df, CONFIG)

    # Task 3: LSTM Price Model
    lstm_price, price_scaler = build_lstm_price_model(df, CONFIG)

    # Task 4: LSTM Volatility Model
    lstm_vol, vol_scaler = build_lstm_volatility_model(df, CONFIG)

    # Task 6: Ensemble signal
    signal = run_ensemble_prediction(
        df, ml_clf, ml_reg, scaler_d, feat_d,
        lstm_price, price_scaler,
        lstm_vol,   vol_scaler,
        CONFIG
    )
    print_ensemble_signal(signal)

    # Task 7: Final dashboard
    build_final_dashboard(df, signal, CONFIG)

    # Task 8: Save
    save_phase_e_outputs(
        lstm_price, price_scaler,
        lstm_vol,   vol_scaler,
        df, signal
    )


def mode_signal():
    """Load saved models, generate one signal."""
    print("=" * 58)
    print("  PHASE E — SIGNAL MODE")
    print("=" * 58)

    # Load Phase D ML models
    try:
        from config_e import (PHASE_D_CLF, PHASE_D_REG,
                               PHASE_D_SCALER, PHASE_D_FEATURES)
        ml_clf   = joblib.load(PHASE_D_CLF)
        ml_reg   = joblib.load(PHASE_D_REG)
        scaler_d = joblib.load(PHASE_D_SCALER)
        with open(PHASE_D_FEATURES) as f:
            feat_d = json.load(f)
    except FileNotFoundError:
        print("❌ Phase D models not found. Run Phase D --mode train first!")
        return

    # Load LSTM if available
    lstm_price, price_scaler = None, None
    lstm_vol, vol_scaler = None, None
    try:
        from tensorflow.keras.models import load_model
        import joblib as jb
        lstm_price   = load_model(str(MODELS_DIR_E / "lstm_price_model.keras"))
        lstm_vol     = load_model(str(MODELS_DIR_E / "lstm_vol_model.keras"))
        price_scaler = jb.load(MODELS_DIR_E / "price_scaler_e.pkl")
        vol_scaler   = jb.load(MODELS_DIR_E / "vol_scaler_e.pkl")
        print("✅ LSTM models loaded.")
    except Exception:
        print("⚠️  LSTM models not found — using ML only.")

    # Load + process data
    df, _, _, _, _, _ = load_all_phases()
    df = add_session_signals(df, CONFIG)

    signal = run_ensemble_prediction(
        df, ml_clf, ml_reg, scaler_d, feat_d,
        lstm_price, price_scaler,
        lstm_vol, vol_scaler,
        CONFIG
    )
    print_ensemble_signal(signal)
    build_final_dashboard(df, signal, CONFIG)
    

def mode_live():
    import sys
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.image as mpimg
    from datetime import datetime
    from config_e import IST, BASE_DIR_E, PLOTS_DIR_E

    phase_d_path = str(BASE_DIR_E.parent / "phase_d")
    if phase_d_path not in sys.path:
        sys.path.insert(0, phase_d_path)

    try:
        from config_e import (PHASE_D_CLF, PHASE_D_REG,
                               PHASE_D_SCALER, PHASE_D_FEATURES)
        ml_clf   = joblib.load(PHASE_D_CLF)
        ml_reg   = joblib.load(PHASE_D_REG)
        scaler_d = joblib.load(PHASE_D_SCALER)
        with open(PHASE_D_FEATURES) as f:
            feat_d = json.load(f)
    except FileNotFoundError:
        print("Phase D models not found!")
        return

    lstm_price = lstm_vol = price_scaler = vol_scaler = None
    try:
        from tensorflow.keras.models import load_model
        import joblib as jb
        lstm_price   = load_model(str(MODELS_DIR_E / "lstm_price_model.keras"))
        lstm_vol     = load_model(str(MODELS_DIR_E / "lstm_vol_model.keras"))
        price_scaler = jb.load(MODELS_DIR_E / "price_scaler_e.pkl")
        vol_scaler   = jb.load(MODELS_DIR_E / "vol_scaler_e.pkl")
    except Exception:
        pass

    print("🚀 PHASE E LIVE SYSTEM STARTED!")
    print(f"📡 Refreshing every {CONFIG['refresh_seconds']}s")
    print("Press Ctrl+C to stop\n")

    plt.ion()
    fig = plt.figure(figsize=(18, 14), facecolor="#1e1e2e")
    plt.show(block=False)

    def update(frame):
        ts = datetime.now(IST).strftime("%H:%M:%S")
        print(f"\r[{ts}] Updating...", end="")

        from data_pipeline import load_or_download_intraday
        from indicators_d  import add_intraday_indicators
        from signals_d     import add_intraday_signals

        try:
            import yfinance as yf
            import pandas as pd
            raw = yf.download(
                CONFIG["ticker"], period="5d",
                interval="5m", auto_adjust=True,
                progress=False
            )
            if raw.empty:
                return
            if hasattr(raw.columns, "levels"):
                raw.columns = raw.columns.get_level_values(0)
            raw.index = raw.index.tz_convert(IST)
            raw = raw.between_time("09:15", "15:30")
            raw.reset_index(inplace=True)
            dt = raw.columns[0]
            raw.rename(columns={dt: "Datetime"}, inplace=True)
            raw["Date"]    = raw["Datetime"].dt.date
            raw["Time"]    = raw["Datetime"].dt.time
            raw["Day_of_Week"]        = raw["Datetime"].dt.dayofweek
            raw["Minutes_Since_Open"] = (
                raw["Datetime"].dt.hour * 60 +
                raw["Datetime"].dt.minute - 555
            )
            df_live = add_intraday_indicators(raw, CONFIG)
            df_live = add_intraday_signals(df_live, CONFIG)
            df_live = add_session_signals(df_live, CONFIG)
        except Exception as e:
            print(f"  Fetch error: {e}")
            return

        sig = run_ensemble_prediction(
            df_live, ml_clf, ml_reg, scaler_d, feat_d,
            lstm_price, price_scaler,
            lstm_vol, vol_scaler, CONFIG
        )

        build_final_dashboard(df_live, sig, CONFIG)

        try:
            img_path = str(PLOTS_DIR_E / "final_dashboard.png")
            img = mpimg.imread(img_path)
            fig.clear()
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis("off")
            ax.set_position([0, 0, 1, 1])
            fig.canvas.draw()
            fig.canvas.flush_events()
            print(f"  Signal={sig.get('signal','?')} "
                  f"Conf={sig.get('ensemble_confidence',0):.1f}%")
        except Exception as e:
            print(f"  Display error: {e}")

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=CONFIG["refresh_seconds"] * 1000,
        cache_frame_data=False
    )

    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("\n⛔ Live system stopped.")
# ── Argument Parser ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase E — Final LSTM Ensemble System"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "signal", "live"],
        default="train",
        help="train | signal | live"
    )
    args = parser.parse_args()

    if   args.mode == "train":  mode_train()
    elif args.mode == "signal": mode_signal()
    elif args.mode == "live":   mode_live()