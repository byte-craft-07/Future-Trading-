# =============================================================
# FILE: trading_system/phase_e/ensemble_engine.py
# PURPOSE: Task 6 — Weighted ensemble of all models + signal
# =============================================================

import json
import numpy as np
import pandas as pd
from datetime import datetime
from config_e import CONFIG, IST, SIGNALS_DIR_E
from lstm_models import lstm_predict_latest
from sentiment_engine import fetch_news_sentiment


# ── Phase E signal names ──────────────────────────────────────
PHASE_E_SIGNAL_NAMES = [
    "E1_KZ_Sweep_Reversal",  "E2_Morning_KZ",
    "E3_Afternoon_KZ",       "E4_PDH_Sweep",
    "E5_PDL_Sweep",          "E6_Weekly_High_Sweep",
    "E7_Weekly_Low_Sweep",   "E8_Sweep_Shift",
    "E9_Double_Sweep",       "E10_Demand_Sweep",
    "E11_Supply_Sweep",      "E12_Liq_Pool_Trap",
    "E13_Stop_Hunt",         "E14_FVG",
    "E15_Reclaim",           "E16_Rejection_Candle",
    "E17_Breakout_Sweep",    "E18_Midpoint",
    "E19_Institutional_Rev", "E20_Institutional_Cont",
    "E21_HQ_Confluence",     "E22_Level_Retest",
    "E23_Daily_Liq",         "E24_Session_Bias",
    "E25_No_Rejection_Cont", "E26_RR_Filter",
    "E27_OB_Sweep",          "E28_HV_Reversal",
    "E29_Sweep_Pullback",    "E30_Expert_Rev"
]


def run_ensemble_prediction(df: pd.DataFrame,
                             ml_clf,
                             ml_reg,
                             scaler_d,
                             feat_cols_d: list,
                             lstm_price_model,
                             price_scaler,
                             lstm_vol_model,
                             vol_scaler,
                             config: dict) -> dict:
    """
    Sab models ke predictions ko weighted ensemble mein combine.

    Components:
      ML Classifier  → direction probability (35%)
      LSTM Price     → price direction confidence (35%)
      LSTM Volatility→ volatility-based confidence (15%)
      News Sentiment → market mood score (15%)

    Args:
        df              : Latest intraday DataFrame.
        ml_clf, ml_reg  : Phase D ML models.
        scaler_d        : Phase D scaler.
        feat_cols_d     : Phase D feature columns.
        lstm_*          : LSTM models + scalers.
        config          : CONFIG dict.

    Returns:
        Complete ensemble signal dict.
    """
    latest = df.iloc[-1]
    curr_price = float(latest["Close"])
    atr_val    = float(latest.get("ATR_14", 1.0))

    # ─────────────────────────────────────────────────────────
    # COMPONENT 1: ML Classifier (Phase D model)
    # ─────────────────────────────────────────────────────────
    avail_d = [f for f in feat_cols_d if f in df.columns]
    row_data = []
    for f in avail_d:
        v = latest.get(f, 0)
        row_data.append(0.0 if pd.isna(v) else float(v))

    X_ml = np.array(row_data).reshape(1, -1)
    if X_ml.shape[1] < len(feat_cols_d):
        pad  = np.zeros((1, len(feat_cols_d) - X_ml.shape[1]))
        X_ml = np.hstack([X_ml, pad])
    else:
        X_ml = X_ml[:, :len(feat_cols_d)]

    try:
        X_sc = scaler_d.transform(X_ml)
        ml_proba  = ml_clf.predict_proba(X_sc)[0]
        ml_buy    = float(ml_proba[1])
        ml_pred_p = float(ml_reg.predict(X_sc)[0])
    except Exception:
        ml_buy    = 0.5
        ml_pred_p = curr_price

    # ─────────────────────────────────────────────────────────
    # COMPONENT 2: LSTM Price (direction from predicted price)
    # ─────────────────────────────────────────────────────────
    lstm_price_feats = ["Close", "VWAP", "EMA_9", "EMA_21",
                         "RSI_14", "MACD", "ATR_14",
                         "Volume_MA_20", "Master_Score"]
    lstm_pred_p = lstm_predict_latest(
        lstm_price_model, price_scaler, df,
        lstm_price_feats, config["sequence_length"]
    )
    if lstm_pred_p == 0.0:
        lstm_pred_p = ml_pred_p

    # Convert to probability: sigmoid of price change
    lstm_price_prob = 1 / (1 + np.exp(
        -(lstm_pred_p - curr_price) / (atr_val + 1e-6)
    ))

    # ─────────────────────────────────────────────────────────
    # COMPONENT 3: LSTM Volatility (confidence adjustment)
    # ─────────────────────────────────────────────────────────
    lstm_vol_feats = ["ATR_14", "BB_Upper", "BB_Lower",
                       "Volume_MA_20", "RSI_14"]
    pred_atr = lstm_predict_latest(
        lstm_vol_model, vol_scaler, df,
        lstm_vol_feats, config["sequence_length"]
    )
    # Low predicted volatility = higher confidence
    curr_atr = atr_val if atr_val > 0 else 1.0
    vol_conf = 1.0 - min(pred_atr / (curr_atr * 2 + 1e-6), 1.0) \
               if pred_atr > 0 else 0.5

    # ─────────────────────────────────────────────────────────
    # COMPONENT 4: News Sentiment
    # ─────────────────────────────────────────────────────────
    sentiment = fetch_news_sentiment(config)
    sent_score = (sentiment["score"] + 1) / 2   # Normalize 0-1

    # ─────────────────────────────────────────────────────────
    # WEIGHTED ENSEMBLE COMBINATION
    # ─────────────────────────────────────────────────────────
    w_ml   = config["weight_ml"]           # 0.35
    w_lp   = config["weight_lstm_price"]   # 0.35
    w_lv   = config["weight_lstm_vol"]     # 0.15
    w_sent = config["weight_sentiment"]    # 0.15

    ensemble_prob = (
        w_ml   * ml_buy          +
        w_lp   * lstm_price_prob +
        w_lv   * vol_conf        +
        w_sent * sent_score
    )
    ensemble_prob = float(np.clip(ensemble_prob, 0.01, 0.99))

    # ── Signal string ──
    if   ensemble_prob >= config["strong_buy_thresh"] : sig_str = "STRONG BUY"
    elif ensemble_prob >= config["buy_thresh"]         : sig_str = "BUY"
    elif ensemble_prob <= config["strong_sell_thresh"] : sig_str = "STRONG SELL"
    elif ensemble_prob <= config["sell_thresh"]        : sig_str = "SELL"
    else                                               : sig_str = "NEUTRAL"

    direction = "UP" if ensemble_prob > 0.5 else "DOWN"

    # ── Final predicted price (ensemble of both) ──
    final_pred = (w_ml * ml_pred_p + w_lp * lstm_pred_p) / (w_ml + w_lp)

    # ── Active E-signals ──
    active_e = [s for s in PHASE_E_SIGNAL_NAMES
                if s in df.columns and latest.get(s, 0) == 1]

    # ── Build final signal ──
    signal = {
        "timestamp"           : datetime.now(IST).strftime(
                                 "%Y-%m-%d %H:%M:%S IST"),
        "ticker"              : config["ticker"],
        "current_price"       : round(curr_price, 2),
        "predicted_price"     : round(final_pred, 2),
        "predicted_in"        : "10 minutes",
        "direction"           : direction,
        "signal"              : sig_str,
        "ensemble_confidence" : round(ensemble_prob * 100, 1),

        # Component breakdown
        "ml_probability"      : round(ml_buy * 100, 1),
        "lstm_price_prob"     : round(lstm_price_prob * 100, 1),
        "lstm_price_predicted": round(lstm_pred_p, 2),
        "vol_confidence"      : round(vol_conf * 100, 1),
        "sentiment_score"     : sentiment["score"],
        "sentiment_signal"    : sentiment["signal"],

        # Signals
        "master_score"        : int(latest.get("Master_Score", 0)),
        "session_strong"      : int(latest.get("Session_Strong", 0)),
        "expert_signal"       : int(latest.get("E30_Expert_Rev", 0)),
        "active_e_signals"    : active_e,

        # Risk management
        "stop_loss"           : round(final_pred - 2 * atr_val, 2),
        "take_profit"         : round(final_pred + 4 * atr_val, 2),
        "risk_reward"         : "1:2",
        "atr"                 : round(atr_val, 2),

        # Market context
        "rsi"                 : round(float(latest.get("RSI_14", 50)), 1),
        "vwap"                : round(float(latest.get("VWAP", curr_price)), 2),
        "above_vwap"          : bool(curr_price > float(
                                     latest.get("VWAP", curr_price))),
        "orb_status"          : ("Above" if curr_price >
                                  float(latest.get("ORB_High", 0))
                                  else "Below"),
    }

    # Save signal JSON
    try:
        p = SIGNALS_DIR_E / "ensemble_signal.json"
        with open(p, "w") as f:
            json.dump(signal, f, indent=2, default=str)
    except Exception as e:
        print(f"⚠️  Signal save error: {e}")

    return signal


def print_ensemble_signal(signal: dict):
    """Print full ensemble signal in terminal."""
    emoji = {"STRONG BUY": "🚀", "BUY": "📈",
             "STRONG SELL": "🔻", "SELL": "📉",
             "NEUTRAL": "⚖️"}.get(signal["signal"], "❓")

    print("\n" + "╔" + "═" * 50 + "╗")
    print(f"║  {emoji} FINAL SIGNAL: {signal['signal']:<36}║")
    print("╠" + "═" * 50 + "╣")
    print(f"║  📈 Current  : ₹{signal['current_price']:>12,.2f}              ║")
    print(f"║  🔮 Predicted: ₹{signal['predicted_price']:>12,.2f} (10 min)    ║")
    print(f"║  💪 Ensemble : {signal['ensemble_confidence']:>5.1f}% confidence           ║")
    print("╠" + "─" * 50 + "╣")
    print(f"║  🤖 ML Model : {signal['ml_probability']:>5.1f}%                          ║")
    print(f"║  🧠 LSTM     : {signal['lstm_price_prob']:>5.1f}%                          ║")
    print(f"║  📊 Vol Conf : {signal['vol_confidence']:>5.1f}%                          ║")
    print(f"║  💬 Sentiment: {signal['sentiment_signal']:<36}║")
    print("╠" + "─" * 50 + "╣")
    print(f"║  🎯 Master Score  : {signal['master_score']:>3}/78 signals active   ║")
    e30 = "✅ ACTIVE" if signal["expert_signal"] else "  inactive"
    print(f"║  ⭐ E30 Expert    : {e30:<32}║")
    print(f"║  🛑 Stop Loss     : ₹{signal['stop_loss']:>12,.2f}              ║")
    print(f"║  🎯 Take Profit   : ₹{signal['take_profit']:>12,.2f}              ║")
    print("╚" + "═" * 50 + "╝")
    if signal["active_e_signals"]:
        print(f"  E-Signals: {', '.join(signal['active_e_signals'][:4])}")

# EXPLANATION: Ensemble = sab models ki "vote".
# ML 35% + LSTM price 35% + volatility 15% +
# sentiment 15% = 100%. Jab sab agree karein
# to strong signal. Ek model galat bhi ho to
# baaki compensate kar dete hain.