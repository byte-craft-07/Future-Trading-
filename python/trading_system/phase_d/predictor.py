# =============================================================
# FILE: trading_system/phase_d/predictor.py
# PURPOSE: Task 4 — Classifier (UP/DOWN) + Regressor (price)
#          10-minute ahead prediction
# =============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                               GradientBoostingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                              mean_absolute_error)
from config_d import CONFIG, INTRADAY_SIGNALS


def build_prediction_models(df: pd.DataFrame, config: dict):
    """
    Two models train karo:
      Classifier → UP ya DOWN (10 min ahead)
      Regressor  → Actual price (10 min ahead)

    Args:
        df    : Full intraday featured DataFrame.
        config: CONFIG dict.

    Returns:
        classifier, regressor, scaler, feature_cols
    """
    print("\n" + "─" * 55)
    print("🤖 BUILDING 10-MIN PREDICTION MODELS")
    print("─" * 55)

    # ── PART A: Classifier target ──
    # 2 candles ahead = 10 min
    n_ahead = config["predict_candles"]
    df = df.copy()
    df["Target_Direction"] = (
        df["Close"].shift(-n_ahead) > df["Close"]
    ).astype(int)

    # ── PART B: Regressor target ──
    df["Target_Price"] = df["Close"].shift(-n_ahead)

    # Drop last n_ahead rows (no target available)
    df = df.iloc[:-n_ahead].copy()

    # ── Feature columns ──
    exclude = ["Datetime", "Date", "Time",
               "Target_Direction", "Target_Price",
               "Open", "High", "Low", "Close", "Volume"]
    avail_signals = [s for s in INTRADAY_SIGNALS if s in df.columns]

    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and df[c].dtype in [np.float64, np.int64, np.int32]
                    and df[c].isnull().sum() == 0]

    print(f"Feature columns   : {len(feature_cols)}")
    print(f"Intraday signals  : {len(avail_signals)}")

    X = df[feature_cols].values
    y_cls = df["Target_Direction"].values
    y_reg = df["Target_Price"].values

    # ── Time-series split (no shuffle) ──
    split = int(len(X) * (1 - config["test_size"]))
    Xtr, Xte        = X[:split], X[split:]
    y_cls_tr, y_cls_te = y_cls[:split], y_cls[split:]
    y_reg_tr, y_reg_te = y_reg[:split], y_reg[split:]

    print(f"Train: {split}  |  Test: {len(X)-split}")

    # ── Scale ──
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(Xtr)
    Xte_s  = scaler.transform(Xte)

    # ── Train Classifier ──
    print("\nTraining Direction Classifier (UP/DOWN)...")
    clf = GradientBoostingClassifier(
        n_estimators  = 100,
        max_depth     = 4,
        learning_rate = 0.05,
        subsample     = 0.8,
        random_state  = config["random_state"]
    )
    clf.fit(Xtr_s, y_cls_tr)

    pred_cls = clf.predict(Xte_s)
    acc      = accuracy_score(y_cls_te, pred_cls)
    f1       = f1_score(y_cls_te, pred_cls, average="weighted")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  F1 Score : {f1:.4f}")

    # ── Train Regressor ──
    print("\nTraining Price Regressor (₹ value)...")
    reg = GradientBoostingRegressor(
        n_estimators  = 100,
        max_depth     = 4,
        learning_rate = 0.05,
        subsample     = 0.8,
        random_state  = config["random_state"]
    )
    reg.fit(Xtr_s, y_reg_tr)

    pred_reg = reg.predict(Xte_s)
    mae      = mean_absolute_error(y_reg_te, pred_reg)
    mape     = np.mean(np.abs((y_reg_te - pred_reg)
                               / y_reg_te)) * 100
    rmse     = np.sqrt(np.mean((y_reg_te - pred_reg)**2))

    print(f"  MAE      : ₹{mae:.2f}")
    print(f"  RMSE     : ₹{rmse:.2f}")
    print(f"  MAPE     : {mape:.2f}%")
    print(f"  Avg prediction error: {mape:.2f}%")

    return clf, reg, scaler, feature_cols

# EXPLANATION: Do models hain:
# Classifier = UP ya DOWN 10 min mein (binary)
# Regressor  = exact price kitna hoga (continuous)
# Dono saath use karenge — "10 min mein ₹2450 hoga
# aur direction UP hai (71% confidence)".
# shift(-2) from future — leakage nahi, yeh future
# candles hain jo ab nahi hain.