# =============================================================
# FILE: trading_system/phase_a/target.py
# PURPOSE: Task 5 — Binary target label banao (UP=1, DOWN=0)
# =============================================================

import pandas as pd    # DataFrame operations


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Next day price upar gayi → 1 (UP)
    Next day price neeche aayi → 0 (DOWN)

    Args:
        df: Featured DataFrame.

    Returns:
        DataFrame with 'Target' column added.
    """
    print("\n" + "─" * 50)
    print("🎯 TARGET LABEL CREATION")
    print("─" * 50)

    # shift(-1) = agle din ka Close aaj ki row mein laao
    # Agar agle din ka Close > aaj ka Close → 1, warna 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Last row drop karo — uske paas koi "next day" nahi
    df = df.iloc[:-1].reset_index(drop=True)

    # ── Class Distribution ──
    counts = df["Target"].value_counts().sort_index()
    total  = len(df)
    print(f"Total samples  : {total}")
    print(f"DOWN (0) days  : {counts.get(0,0)}  ({counts.get(0,0)/total*100:.1f}%)")
    print(f"UP   (1) days  : {counts.get(1,0)}  ({counts.get(1,0)/total*100:.1f}%)")

    # Balance check
    ratio = counts.get(1, 0) / total
    if 0.45 <= ratio <= 0.55:
        print("✅ Dataset well-balanced (near 50/50)")
    else:
        print("⚠️ Imbalanced — consider class_weight='balanced'")

    return df