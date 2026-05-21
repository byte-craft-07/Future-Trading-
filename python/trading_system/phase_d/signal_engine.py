# =============================================================
# FILE: trading_system/phase_d/signal_engine.py
# PURPOSE: Task 5 — TradingView-style signal generate karo
# =============================================================

import json
import numpy as np
import pandas as pd
from datetime import datetime
from config_d import CONFIG, IST, SIGNALS_DIR, INTRADAY_SIGNALS


def generate_signal(df: pd.DataFrame,
                    clf, reg, scaler,
                    feature_cols: list,
                    config: dict) -> dict:
    """
    Latest candle se 10-min ahead signal generate karo.
    Direction + Price + Confidence + JSON output.

    Args:
        df          : Full intraday DataFrame.
        clf         : Trained direction classifier.
        reg         : Trained price regressor.
        scaler      : Fitted StandardScaler.
        feature_cols: Feature column list.
        config      : CONFIG dict.

    Returns:
        Signal dict (TradingView-style).
    """
    # ── Latest row ──
    latest = df.iloc[-1]
    avail  = [f for f in feature_cols if f in df.columns]

    # Handle missing features
    row_data = []
    for f in avail:
        val = latest.get(f, 0)
        row_data.append(0 if pd.isna(val) else val)

    X_live = np.array(row_data).reshape(1, -1)

    # Handle feature count mismatch
    if X_live.shape[1] != len(feature_cols):
        # Pad or trim
        if X_live.shape[1] < len(feature_cols):
            pad = np.zeros((1, len(feature_cols) - X_live.shape[1]))
            X_live = np.hstack([X_live, pad])
        else:
            X_live = X_live[:, :len(feature_cols)]

    try:
        X_scaled = scaler.transform(X_live)
    except Exception:
        X_scaled = X_live

    # ── Predictions ──
    try:
        proba          = clf.predict_proba(X_scaled)[0]
        buy_prob       = float(proba[1])
        sell_prob      = float(proba[0])
        pred_price     = float(reg.predict(X_scaled)[0])
    except Exception as e:
        print(f"⚠️  Prediction error: {e}")
        buy_prob, sell_prob = 0.5, 0.5
        pred_price = float(latest["Close"])

    current_price = float(latest["Close"])
    atr_val       = float(latest.get("ATR_14", 0))

    # ── Signal strength ──
    if   buy_prob  >= 0.65: signal_str = "STRONG BUY"
    elif buy_prob  >= 0.55: signal_str = "BUY"
    elif sell_prob >= 0.65: signal_str = "STRONG SELL"
    elif sell_prob >= 0.55: signal_str = "SELL"
    else:                   signal_str = "NEUTRAL"

    direction = "UP" if buy_prob > sell_prob else "DOWN"

    # ── Active strategy signals ──
    active = [s for s in INTRADAY_SIGNALS
              if s in df.columns and latest.get(s, 0) == 1]

    # ── Build signal dict ──
    signal = {
        "timestamp"        : datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "ticker"           : config["ticker"],
        "current_price"    : round(current_price, 2),
        "predicted_price"  : round(pred_price, 2),
        "predicted_in"     : "10 minutes",
        "direction"        : direction,
        "signal"           : signal_str,
        "confidence"       : round(buy_prob * 100, 1),
        "active_strategies": active,
        "confluence_score" : int(latest.get("Total_Score", 0)),
        "rsi"              : round(float(latest.get("RSI_14", 0)), 1),
        "vwap"             : round(float(latest.get("VWAP", 0)), 2),
        "above_vwap"       : bool(latest.get("Close", 0) >
                                   latest.get("VWAP", 0)),
        "orb_status"       : ("Above" if current_price >
                               float(latest.get("ORB_High", 0))
                               else "Below"),
        "stop_loss"        : round(pred_price - 2 * atr_val, 2),
        "take_profit"      : round(pred_price + 4 * atr_val, 2),
        "risk_reward"      : "1:2",
    }

    # ── Save JSON ──
    json_path = SIGNALS_DIR / "latest_signal.json"
    try:
        with open(json_path, "w") as f:
            json.dump(signal, f, indent=2, default=str)
    except Exception as e:
        print(f"⚠️  Signal save failed: {e}")

    return signal


def print_signal_box(signal: dict):
    """Print signal in clean box format."""
    emoji = {"STRONG BUY": "🚀", "BUY": "📈",
             "STRONG SELL": "🔻", "SELL": "📉",
             "NEUTRAL": "⚖️"}.get(signal["signal"], "❓")

    conf_color = "💪" if signal["confidence"] >= 65 else "📊"

    print("\n" + "┌" + "─" * 44 + "┐")
    print(f"│  {emoji} SIGNAL: {signal['signal']:<33}│")
    print(f"│  📈 Current  : ₹{signal['current_price']:>10,.2f}              │")
    print(f"│  🔮 Predicted: ₹{signal['predicted_price']:>10,.2f} (10 min)     │")
    print(f"│  {conf_color} Confidence: {signal['confidence']:>5.1f}%                   │")
    print(f"│  📊 Confluence:{signal['confluence_score']:>3}/25 signals active   │")
    print(f"│  📍 RSI      : {signal['rsi']:>5.1f}                         │")
    print(f"│  💧 VWAP     : ₹{signal['vwap']:>10,.2f}              │")
    print(f"│  🛑 Stop Loss: ₹{signal['stop_loss']:>10,.2f}              │")
    print(f"│  🎯 Target   : ₹{signal['take_profit']:>10,.2f}              │")
    print(f"│  ⚡ ORB      : {signal['orb_status']:<31}│")
    print("└" + "─" * 44 + "┘")
    if signal["active_strategies"]:
        print(f"  Active signals: {', '.join(signal['active_strategies'][:5])}")

# EXPLANATION: Signal engine latest candle ka data
# dono models ko deta hai. Classifier bata deta hai
# UP ya DOWN probability. Regressor actual ₹ price
# predict karta hai. SL aur TP ATR se auto-calculate
# hote hain. JSON mein save hota hai — isko aage
# Telegram bot ya dashboard mein use kar sakte hain.