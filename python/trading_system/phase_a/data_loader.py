# =============================================================
# FILE: trading_system/phase_a/data_loader.py
# PURPOSE: Task 2 — Download karo ya CSV se load karo
# =============================================================

import yfinance as yf          # Free stock data download
import pandas as pd            # DataFrame operations
from config import CONFIG, DATA_RAW


def download_data(config: dict) -> pd.DataFrame:
    """
    Yahoo Finance se historical OHLCV data download karo.

    Args:
        config: CONFIG dictionary (ticker, dates, interval).

    Returns:
        Raw OHLCV DataFrame.
    """
    print(f"\n📡 Downloading {config['ticker']} from Yahoo Finance...")
    try:
        df = yf.download(
            tickers     = config["ticker"],
            start       = config["start_date"],
            end         = config["end_date"],
            interval    = config["interval"],
            progress    = False,
            auto_adjust = False
        )

        # MultiIndex columns ko flatten karo (yfinance sometimes deta hai)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Date ko index se column mein lao
        df.reset_index(inplace=True)

        print(f"✅ Downloaded {len(df)} rows.")
        return df

    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise


def load_or_download(config: dict) -> pd.DataFrame:
    """
    Offline mode: CSV already hai → load karo.
    Online mode : CSV nahi hai → download + save karo.

    Args:
        config: CONFIG dictionary.

    Returns:
        Raw OHLCV DataFrame.
    """
    # CSV file ka path banao
    raw_csv = DATA_RAW / f"{config['ticker'].replace('.', '_')}_raw.csv"

    if raw_csv.exists():
        # ── Offline Mode ──
        print(f"\n📂 [OFFLINE MODE] Loading: {raw_csv}")
        try:
            df = pd.read_csv(raw_csv, parse_dates=["Date"])
            print(f"✅ Loaded {len(df)} rows from local CSV.")
        except Exception as e:
            print(f"❌ CSV load failed: {e}. Re-downloading...")
            df = download_data(config)
            df.to_csv(raw_csv, index=False)
    else:
        # ── Online Mode ──
        print(f"\n🌐 [ONLINE MODE] CSV not found. Downloading...")
        df = download_data(config)
        try:
            df.to_csv(raw_csv, index=False)
            print(f"💾 Saved to: {raw_csv}")
        except Exception as e:
            print(f"⚠️ Could not save CSV: {e}")

    # ── Data Summary Print ──
    print("\n" + "─" * 50)
    print("📊 RAW DATA SUMMARY")
    print("─" * 50)
    print(f"Shape      : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Date Range : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"Columns    : {list(df.columns)}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nFirst 5 Rows:\n{df.head()}")

    return df