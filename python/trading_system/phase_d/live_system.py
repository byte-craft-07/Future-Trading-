# =============================================================
# FILE: trading_system/phase_d/live_system.py
# PURPOSE: Task 7 — Auto-refresh live system (60 sec update)
# =============================================================

import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pytz
from config_d import CONFIG, IST
from data_pipeline   import load_or_download_intraday
from indicators_d    import add_intraday_indicators
from signals_d       import add_intraday_signals
from signal_engine   import generate_signal, print_signal_box
from dashboard       import build_live_dashboard


def is_market_open(config: dict) -> bool:
    """
    NSE market hours check karo (09:15 - 15:30 IST).

    Returns:
        True if market is open, False otherwise.
    """
    now = datetime.now(IST)

    # Weekend check
    if now.weekday() >= 5:
        return False

    # Time check
    t = now.strftime("%H:%M")
    return config["market_open"] <= t <= config["market_close"]


def fetch_and_process(config: dict):
    """
    Latest data download + all features add karo.
    Live mode mein CSV bypass karke fresh data lena zaroori hai.

    Args:
        config: CONFIG dict.

    Returns:
        Processed DataFrame ya None on failure.
    """
    import yfinance as yf
    import pandas as pd

    try:
        raw = yf.download(
            tickers     = config["ticker"],
            period      = "5d",           # Last 5 days for speed
            interval    = config["interval"],
            auto_adjust = True,
            progress    = False
        )
        if raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.index = raw.index.tz_convert(IST)
        raw = raw.between_time(config["market_open"],
                               config["market_close"])
        raw.reset_index(inplace=True)
        dt_col = raw.columns[0]
        raw.rename(columns={dt_col: "Datetime"}, inplace=True)

        raw["Date"]     = raw["Datetime"].dt.date
        raw["Time"]     = raw["Datetime"].dt.time
        raw["Day_of_Week"]       = raw["Datetime"].dt.dayofweek
        market_open_min = 9 * 60 + 15
        raw["Minutes_Since_Open"] = (
            raw["Datetime"].dt.hour * 60 +
            raw["Datetime"].dt.minute - market_open_min
        )

        df = add_intraday_indicators(raw, config)
        df = add_intraday_signals(df, config)
        return df

    except Exception as e:
        print(f"⚠️  Fetch error: {e}")
        return None


def run_live_system(clf, reg, scaler,
                    feature_cols: list,
                    config: dict):
    """
    Live auto-refreshing system start karo.
    Har 60 seconds mein data refresh hoga.
    Chart + signal update hoga automatically.

    Args:
        clf, reg     : Trained models.
        scaler       : Fitted scaler.
        feature_cols : Feature column names.
        config       : CONFIG dict.
    """
    print("\n" + "═" * 55)
    print("🚀 Live System Started!")
    print(f"📡 Refreshing every {config['refresh_seconds']} seconds")
    print(f"⏰ Market Hours: {config['market_open']} - "
          f"{config['market_close']} IST")
    print("Press Ctrl+C to stop")
    print("═" * 55)

    # Use non-interactive backend for animation
    plt.ion()
    fig = plt.figure(figsize=(16, 12), facecolor="#1e1e2e")
    plt.show(block=False)

    frame_count = [0]

    def update_chart(frame):
        """Animation callback — runs every refresh_seconds."""
        frame_count[0] += 1
        ts = datetime.now(IST).strftime("%H:%M:%S")

        if is_market_open(config):
            print(f"\r[{ts}] 📡 Fetching live data... "
                  f"(frame #{frame_count[0]})", end="")
            df_live = fetch_and_process(config)
        else:
            print(f"\n[{ts}] 🔴 Market closed. Showing last data.")
            df_live = fetch_and_process(config)

        if df_live is None or len(df_live) < 30:
            print("  ⚠️  Not enough data.")
            return

        # Generate signal
        signal = generate_signal(
            df_live, clf, reg, scaler, feature_cols, config
        )

        # Rebuild dashboard on same figure
        fig.clear()
        build_live_dashboard(df_live, signal, config, save=False)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Also print signal to terminal
        if frame_count[0] % 5 == 1:  # Print every 5th frame
            print_signal_box(signal)

    # ── Animation: runs update_chart every refresh_seconds ──
    ani = animation.FuncAnimation(
        fig,
        update_chart,
        interval = config["refresh_seconds"] * 1000,
        cache_frame_data = False
    )

    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("\n\n⛔ Live system stopped by user.")

# EXPLANATION: FuncAnimation se chart har 60 seconds
# mein automatically update hota hai. Market band hone
# pe bhi last data dikhata hai. Ctrl+C se band karo.
# Isko background mein chalate hue aap trading kar
# sakte hain — signal check karte rehna.