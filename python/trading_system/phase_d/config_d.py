# =============================================================
# FILE: trading_system/phase_d/config_d.py
# PURPOSE: Phase D CONFIG + all paths + signal column names
# =============================================================

import pytz
from pathlib import Path

# ── Folder Paths ─────────────────────────────────────────────
BASE_DIR_D  = Path(__file__).parent
PHASE_C_DIR = BASE_DIR_D.parent / "phase_c"

DATA_RAW_D   = BASE_DIR_D / "data" / "raw"
DATA_PROC_D  = BASE_DIR_D / "data" / "processed"
MODELS_DIR_D = BASE_DIR_D / "models"
SIGNALS_DIR  = BASE_DIR_D / "signals"
PLOTS_DIR_D  = BASE_DIR_D / "plots"

for _f in [DATA_RAW_D, DATA_PROC_D, MODELS_DIR_D,
           SIGNALS_DIR, PLOTS_DIR_D]:
    _f.mkdir(parents=True, exist_ok=True)

# ── Phase C references ────────────────────────────────────────
PHASE_C_MODEL      = PHASE_C_DIR / "models" / "best_model_phase_c.pkl"
PHASE_C_SCALER     = PHASE_C_DIR / "models" / "scaler_phase_c.pkl"
PHASE_C_TOP_STRATS = PHASE_C_DIR / "backtest" / "results" / "top_strategies.json"

# ── IST Timezone ──────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")

# ── All 25 intraday signal names ─────────────────────────────
INTRADAY_SIGNALS = [
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

CONFIG = {
    # Data
    "ticker"            : "RELIANCE.NS",
    "interval"          : "5m",
    "period"            : "60d",

    # Market timing (IST)
    "market_open"       : "09:15",
    "market_close"      : "15:30",
    "orb_minutes"       : 30,
    "ist_timezone"      : "Asia/Kolkata",

    # Prediction
    "predict_candles"   : 2,        # 2 candles × 5min = 10 min
    "candle_minutes"    : 5,

    # Indicators
    "rsi_period"        : 14,
    "macd_fast"         : 12,
    "macd_slow"         : 26,
    "macd_signal"       : 9,
    "ema_fast"          : 9,
    "ema_slow"          : 21,
    "bb_period"         : 20,
    "atr_period"        : 14,
    "supertrend_period" : 10,
    "supertrend_mult"   : 3.0,

    # Strategy thresholds
    "rsi_oversold"      : 30,
    "rsi_overbought"    : 70,
    "volume_spike"      : 1.5,
    "vwap_dev_pct"      : 0.5,
    "orb_buffer"        : 0.001,

    # Live system
    "refresh_seconds"   : 60,
    "display_candles"   : 100,
    "signal_threshold"  : 0.60,

    # Model
    "test_size"         : 0.2,
    "random_state"      : 42,
}