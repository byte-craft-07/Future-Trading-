# =============================================================
# FILE: trading_system/phase_a/cleaner.py
# PURPOSE: Task 3 — Galat data hatao, NaN fix karo
# =============================================================

import pandas as pd    # DataFrame operations


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw OHLCV data clean karo:
      - Invalid Close rows hatao
      - Duplicates hatao
      - Baaki NaN forward-fill karo

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    print("\n" + "─" * 50)
    print("🧹 DATA CLEANING REPORT")
    print("─" * 50)

    # ── Problems report karo (before cleaning) ──
    print(f"NaN count per column:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows     : {df.duplicated().sum()}")
    print(f"Rows with Close ≤ 0: {(df['Close'] <= 0).sum()}")

    # ── Invalid Close rows hatao ──
    df = df[df["Close"].notna() & (df["Close"] > 0)]

    # ── Exact duplicate rows hatao ──
    df = df.drop_duplicates()

    # ── Baaki NaN values forward-fill karo ──
    # (last valid value carry forward hota hai)
    df = df.ffill()

    # ── Index reset karo ──
    df = df.reset_index(drop=True)

    print(f"\n✅ Final shape after cleaning: {df.shape}")
    return df