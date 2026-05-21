# =============================================================
# FILE: trading_system/phase_e/config_e.py
# PURPOSE: Phase E CONFIG + all 78-signal names + paths
# =============================================================

import pytz
from pathlib import Path

# ── Folder Paths ─────────────────────────────────────────────
BASE_DIR_E  = Path(__file__).parent
PHASE_D_DIR = BASE_DIR_E.parent / "phase_d"
PHASE_C_DIR = BASE_DIR_E.parent / "phase_c"

DATA_PROC_E  = BASE_DIR_E / "data" / "processed"
MODELS_DIR_E = BASE_DIR_E / "models"
PLOTS_DIR_E  = BASE_DIR_E / "plots"
SIGNALS_DIR_E= BASE_DIR_E / "signals"
LOGS_DIR_E   = BASE_DIR_E / "logs"

for _f in [DATA_PROC_E, MODELS_DIR_E, PLOTS_DIR_E,
           SIGNALS_DIR_E, LOGS_DIR_E]:
    _f.mkdir(parents=True, exist_ok=True)

# ── Phase D Inputs ────────────────────────────────────────────
PHASE_D_CSV      = PHASE_D_DIR / "data" / "processed" / "RELIANCE_5min_features.csv"
PHASE_D_CLF      = PHASE_D_DIR / "models" / "classifier_phase_d.pkl"
PHASE_D_REG      = PHASE_D_DIR / "models" / "regressor_phase_d.pkl"
PHASE_D_SCALER   = PHASE_D_DIR / "models" / "scaler_phase_d.pkl"
PHASE_D_FEATURES = PHASE_D_DIR / "models" / "phase_d_features.json"
PHASE_C_TOP_STRAT= PHASE_C_DIR / "backtest" / "results" / "top_strategies.json"

IST = pytz.timezone("Asia/Kolkata")

# ── Signal Column Groups ──────────────────────────────────────
PHASE_B_SIGNALS = [
    "S1_RSI_MACD_Reversal", "S2_SMA_Crossover",
    "S3_RSI_Pullback",       "S4_MACD_Momentum",
    "S5_Volume_Breakout",    "S6_Engulf_MACD",
    "S7_Oversold_Bounce",    "S8_Trend_Follow",
    "S9_Momentum_Breakout",  "S10_EMA_Cross",
    "S11_Gap_Up",            "S12_Support_Bounce",
    "S13_Pullback_Entry",    "S14_Tiny_Candle",
    "S15_Supertrend"
]
PHASE_C_SIGNALS = [
    "C1_False_Breakout", "C2_False_Trendline",
    "C3_BOS",            "C4_First_Pullback",
    "C5_Retest",         "C6_Range_Expansion",
    "C7_RSI_Divergence", "C8_Structure_Shift"
]
PHASE_D_SIGNALS = [
    "D1_ORB_Breakout",      "D2_ORB_Retest",
    "D3_VWAP_Pullback",     "D4_VWAP_Reversion",
    "D5_VWAP_Trend",        "D6_HTF_Bias",
    "D7_HTF_Trap",          "D8_MTF_BOS",
    "D9_Order_Block",       "D10_Inducement",
    "D11_BOS_Sweep",        "D12_MACD_RSI",
    "D13_TL_Trap",          "D14_Range_Expand",
    "D15_Retest",           "D16_Premium_Discount",
    "D17_Vol_Spike",        "D18_Failed_Breakdown",
    "D19_HTF_Reversal",     "D20_News_Reject",
    "D21_Clean_Confluence", "D22_Breakout_Confirm",
    "D23_Risk_Trend",       "D24_Liq_Sweep",
    "D25_Kill_Zone"
]
PHASE_E_SIGNALS = [f"E{i}" for i in range(1, 31)]
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
ALL_78_SIGNALS = (PHASE_B_SIGNALS + PHASE_C_SIGNALS +
                  PHASE_D_SIGNALS + PHASE_E_SIGNAL_NAMES)

CONFIG = {
    "ticker"              : "RELIANCE.NS",
    "interval"            : "5m",
    "period"              : "60d",
    "ist_timezone"        : "Asia/Kolkata",

    # Live system
    "refresh_seconds"     : 60,
    "display_candles"     : 100,

    # ── Phase D indicator keys (live mode ke liye zaroori) ──
    "rsi_period"          : 14,
    "macd_fast"           : 12,
    "macd_slow"           : 26,
    "macd_signal"         : 9,
    "ema_fast"            : 9,
    "ema_slow"            : 21,
    "bb_period"           : 20,
    "atr_period"          : 14,
    "supertrend_period"   : 10,
    "supertrend_mult"     : 3.0,
    "orb_minutes"         : 30,
    "orb_buffer"          : 0.001,
    "volume_spike"        : 1.5,
    "vwap_dev_pct"        : 0.5,
    "market_open"         : "09:15",
    "market_close"        : "15:30",

    # NSE Session Kill Zones
    "morning_kz_start"    : "09:15",
    "morning_kz_end"      : "09:45",
    "midday_start"        : "11:00",
    "midday_end"          : "13:00",
    "afternoon_kz_start"  : "14:30",
    "afternoon_kz_end"    : "15:30",
    "weekly_period"       : 5,

    # LSTM (CPU-safe settings)
    "sequence_length"     : 20,
    "lstm_epochs"         : 20,
    "lstm_batch"          : 32,
    "lstm_price_units"    : [72, 48],
    "lstm_vol_units"      : [50, 30],
    "dropout_rate"        : 0.2,
    "validation_split"    : 0.1,

    # Sentiment
    "news_sources"        : [
        "https://feeds.feedburner.com/ndtvnews-business",
        "https://economictimes.indiatimes.com/markets/rss.cms"
    ],
    "sentiment_window"    : 3,

    # Ensemble weights
    "weight_ml"           : 0.35,
    "weight_lstm_price"   : 0.35,
    "weight_lstm_vol"     : 0.15,
    "weight_sentiment"    : 0.15,

    # Signal thresholds
    "strong_buy_thresh"   : 0.70,
    "buy_thresh"          : 0.58,
    "sell_thresh"         : 0.42,
    "strong_sell_thresh"  : 0.30,

    # Model
    "test_size"           : 0.2,
    "random_state"        : 42,
    "predict_candles"     : 2,
}