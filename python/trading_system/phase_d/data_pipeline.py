# =============================================================
# FILE: trading_system/phase_d/data_pipeline.py
# PURPOSE: Task 1 — 5min intraday data download + IST filter
# =============================================================

import pytz
import pandas as pd
import yfinance as yf
from config_d import CONFIG, DATA_RAW_D, IST


def download_intraday_data(config: dict) -> pd.DataFrame:
    """
    yfinance se 5-minute OHLCV data download karo.
    IST timezone mein convert karo.
    Market hours (09:15-15:30) filter karo.

    Args:
        config: CONFIG dict.

    Returns:
        Filtered 5min DataFrame in IST.
    """
    print(f"\n📡 Downloading {config['ticker']} "
          f"({config['interval']}, {config['period']})...")
    try:
        raw = yf.download(
            tickers     = config["ticker"],
            period      = config["period"],
            interval    = config["interval"],
            auto_adjust = True,
            progress    = False
        )

        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # ── Convert to IST timezone ──
        if raw.index.tzinfo is None:
            raw.index = raw.index.tz_localize("UTC")
        raw.index = raw.index.tz_convert(IST)

        # ── Filter market hours: 09:15 to 15:30 ──
        raw = raw.between_time(
            config["market_open"],
            config["market_close"]
        )

        raw.reset_index(inplace=True)

        # Rename datetime column
        dt_col = raw.columns[0]
        raw.rename(columns={dt_col: "Datetime"}, inplace=True)

        print(f"✅ Downloaded {len(raw)} candles.")
        return raw

    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise


def load_or_download_intraday(config: dict) -> pd.DataFrame:
    """
    Offline mode: CSV se load karo.
    Online mode : Download + save karo.
    Time-based columns add karo.

    Args:
        config: CONFIG dict.

    Returns:
        Processed DataFrame with Date, Time, etc.
    """
    csv_path = DATA_RAW_D / "RELIANCE_5min_raw.csv"

    if csv_path.exists():
        print(f"\n📂 [OFFLINE] Loading: {csv_path.name}")
        df = pd.read_csv(csv_path, parse_dates=["Datetime"])
        # Re-localize if needed
        if df["Datetime"].dt.tz is None:
            df["Datetime"] = df["Datetime"].dt.tz_localize(IST)
        print(f"✅ Loaded {len(df)} rows.")
    else:
        print(f"\n🌐 [ONLINE] CSV not found. Downloading...")
        df = download_intraday_data(config)
        try:
            df.to_csv(csv_path, index=False)
            print(f"💾 Saved: {csv_path.name}")
        except Exception as e:
            print(f"⚠️  Save failed: {e}")

    # ── Add time-based columns ──
    df["Date"]     = df["Datetime"].dt.date
    df["Time"]     = df["Datetime"].dt.time
    df["Day_of_Week"] = df["Datetime"].dt.dayofweek   # 0=Mon, 4=Fri

    # Minutes since 09:15 each day
    market_open_min = 9 * 60 + 15
    df["Minutes_Since_Open"] = (
        df["Datetime"].dt.hour * 60 +
        df["Datetime"].dt.minute -
        market_open_min
    )

    # ── Print summary ──
    print("\n" + "─" * 50)
    print("📊 INTRADAY DATA SUMMARY")
    print("─" * 50)
    print(f"Shape          : {df.shape}")
    dates = df["Date"].unique()
    print(f"Trading Days   : {len(dates)}")
    print(f"Date Range     : {dates[0]} → {dates[-1]}")
    avg_candles = len(df) / len(dates)
    print(f"Avg Candles/Day: {avg_candles:.0f} (expected ~75)")
    print(f"Missing Values : {df.isnull().sum().sum()}")

    return df

# EXPLANATION: 5min data mein ek din = ~75 candles.
# IST conversion zaroori hai — yfinance UTC mein
# deta hai. Market hours filter karna zaroori hai
# warna pre/post market data bhi aa jaata hai.