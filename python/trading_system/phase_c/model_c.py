# =============================================================
# FILE: trading_system/phase_c/model_c.py
# PURPOSE: Task 8 — Phase C ML model with structure features
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config_c import (CONFIG, PLOTS_DIR_C,
                      PHASE_B_STRATEGIES, PHASE_C_STRATEGIES)


def train_phase_c_model(df: pd.DataFrame,
                        model_b,
                        scaler_b,
                        rank_df: pd.DataFrame):
    """
    Phase C ML model train karo with full 23-signal feature set.
    Top backtested strategies ke signal ko extra feature dena.

    Args:
        df      : Full featured DataFrame.
        model_b : Phase B model (for comparison).
        scaler_b: Phase B scaler (for comparison).
        rank_df : Strategy ranking from backtester.

    Returns:
        model_c, scaler_c, feature_cols_c
    """
    print("\n" + "─" * 58)
    print("🤖 PHASE C MODEL TRAINING")
    print("─" * 58)

    # ── Get top 3 strategies from ranking ──
    top3_strats = []
    if rank_df is not None and len(rank_df) >= 3:
        top3_strats = rank_df["Strategy"].head(3).tolist()
        print(f"Top 3 backtest strategies: {top3_strats}")

    # ── Create Top_Strategy_Active feature ──
    # = 1 if any of the top 3 backtested strategies has signal
    if top3_strats:
        valid_top3 = [s for s in top3_strats if s in df.columns]
        if valid_top3:
            df["Top_Strategy_Active"] = (
                df[valid_top3].sum(axis=1) > 0
            ).astype(int)
        else:
            df["Top_Strategy_Active"] = 0
    else:
        df["Top_Strategy_Active"] = 0

    # ── Phase C feature set ──
    phase_a_feats = [
        "SMA_20", "SMA_50", "EMA_20", "RSI_14",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Mid", "BB_Lower",
        "Volume_MA_20", "Price_vs_SMA20",
        "High_Low_Range", "Body_Size"
    ]
    phase_b_new = [
        "ATR_14", "Supertrend_Signal", "VWAP_Daily",
        "Candle_Body_Ratio", "Bullish_Engulf",
        "Confluence_Score", "Confluence_Strong"
    ]
    phase_c_new = [
        "Confluence_Strong_C",
        "Top_Strategy_Active"
    ]

    all_feat_candidates = (phase_a_feats + phase_b_new +
                           PHASE_B_STRATEGIES +
                           PHASE_C_STRATEGIES +
                           phase_c_new)

    # Only use columns that exist in df
    feature_cols = [f for f in all_feat_candidates
                    if f in df.columns]

    print(f"\nPhase A features    : {len(phase_a_feats)}")
    print(f"Phase B new         : {len(phase_b_new)}")
    print(f"Strategy signals    : "
          f"{len(PHASE_B_STRATEGIES + PHASE_C_STRATEGIES)}")
    print(f"Phase C new         : {len(phase_c_new)}")
    print(f"Total Phase C feats : {len(feature_cols)}")

    X = df[feature_cols].values
    y = df["Target"].values

    # ── Time-series split — NO SHUFFLE ──
    split_idx       = int(len(X) * (1 - CONFIG["test_size"]))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # ── Scale — fit only on train ──
    scaler_c = StandardScaler()
    X_train  = scaler_c.fit_transform(X_train)
    X_test   = scaler_c.transform(X_test)

    # ── Train Gradient Boosting ──
    print("Training Gradient Boosting Classifier...")
    model_c = GradientBoostingClassifier(
        n_estimators  = 100,
        max_depth     = 4,
        learning_rate = 0.05,
        subsample     = 0.8,
        random_state  = CONFIG["random_state"]
    )
    model_c.fit(X_train, y_train)
    print("✅ Phase C model training complete.")

    # ── Evaluate Phase C ──
    pred_c  = model_c.predict(X_test)
    acc_c   = accuracy_score(y_test, pred_c)
    f1_c    = f1_score(y_test, pred_c, average="weighted")

    # ── Compare with Phase B ──
    df_test     = df.iloc[split_idx:].reset_index(drop=True)
    phase_b_raw = [
        "SMA_20", "SMA_50", "EMA_20", "RSI_14",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Mid", "BB_Lower",
        "Volume_MA_20", "Price_vs_SMA20",
        "High_Low_Range", "Body_Size",
        "ATR_14", "Supertrend_Signal", "VWAP_Daily",
        "Candle_Body_Ratio", "Bullish_Engulf",
        "Confluence_Score", "Confluence_Strong"
    ] + PHASE_B_STRATEGIES

    b_feats_avail = [f for f in phase_b_raw if f in df_test.columns]
    try:
        X_test_b = scaler_b.transform(df_test[b_feats_avail].values)
        pred_b   = model_b.predict(X_test_b)
        acc_b    = accuracy_score(y_test, pred_b)
        f1_b     = f1_score(y_test, pred_b, average="weighted")
        print(f"\nPhase B → Acc={acc_b*100:.2f}%  F1={f1_b:.4f}")
    except Exception:
        acc_b, f1_b = 0, 0

    print(f"Phase C → Acc={acc_c*100:.2f}%  F1={f1_c:.4f}")
    improvement = f1_c - f1_b
    if improvement > 0:
        print(f"✅ Phase C improved F1 by +{improvement:.4f}")
    else:
        print(f"ℹ️  F1 change: {improvement:.4f}")

    print(f"\nDetailed Report (Phase C):")
    print(classification_report(y_test, pred_c,
                                target_names=["DOWN", "UP"]))

    # ── Feature Importance Plot ──
    _plot_feature_importance_c(model_c, feature_cols)

    return model_c, scaler_c, feature_cols, df


def _plot_feature_importance_c(model, feature_cols: list):
    """Top 25 features — C1-C8 alag color mein."""
    importances = model.feature_importances_
    top_n       = min(25, len(feature_cols))
    top_idx     = np.argsort(importances)[::-1][:top_n]
    top_feats   = [feature_cols[i] for i in top_idx]
    top_vals    = importances[top_idx]

    def feat_color(name):
        if name.startswith("C") and name[1].isdigit():
            return "darkorange"        # Structure signals
        elif name.startswith("S") and name[1:3].replace("_","").isdigit():
            return "seagreen"          # Phase B strategy signals
        elif name == "Top_Strategy_Active":
            return "crimson"           # Special feature
        else:
            return "steelblue"         # Technical indicators

    colors = [feat_color(f) for f in top_feats]

    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(top_feats[::-1], top_vals[::-1],
                   color=colors[::-1], edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_title(
        "Phase C — Top 25 Feature Importances\n"
        "🟠 Structure (C1-C8)  🟢 Strategy (S1-S15)  "
        "🔴 Top Backtest  🔵 Indicator",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    p = PLOTS_DIR_C / "feature_importance_c.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"📊 Feature importance saved: {p.name}")

# EXPLANATION: Model ab 40+ features dekh ke predict
# karta hai. Top_Strategy_Active = ek special feature
# jo backtesting se proven top strategies ko combine
# karta hai. Agar C signals feature importance mein
# upar hain → structure analysis genuinely kaam kar
# rahi hai model ke liye.