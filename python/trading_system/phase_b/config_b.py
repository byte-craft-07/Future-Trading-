# =============================================================
# FILE: trading_system/phase_b/config_b.py
# PURPOSE: Phase B ka CONFIG + all folder paths
# =============================================================

from pathlib import Path

# ── Folder Paths ─────────────────────────────────────────────
BASE_DIR_B  = Path(__file__).parent            # phase_b/
PHASE_A_DIR = BASE_DIR_B.parent / "phase_a"   # phase_a/

DATA_PROC_B  = BASE_DIR_B / "data" / "processed"
MODELS_DIR_B = BASE_DIR_B / "models"
PLOTS_DIR_B  = BASE_DIR_B / "plots"

# Auto-create folders
for _f in [DATA_PROC_B, MODELS_DIR_B, PLOTS_DIR_B]:
    _f.mkdir(parents=True, exist_ok=True)

# ── Phase A Output Paths ──────────────────────────────────────
PHASE_A_CSV    = PHASE_A_DIR / "data" / "processed" / "RELIANCE_NS_features.csv"
PHASE_A_MODEL  = PHASE_A_DIR / "models" / "best_model_phase_a.pkl"
PHASE_A_SCALER = PHASE_A_DIR / "models" / "scaler_phase_a.pkl"

# ── Phase A Feature List (exactly as trained) ─────────────────
PHASE_A_FEATURES = [
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Mid", "BB_Lower",
    "Volume_MA_20", "Price_vs_SMA20",
    "High_Low_Range", "Body_Size"
]

# ── Strategy + Model Settings ────────────────────────────────
CONFIG = {
    "test_size"         : 0.2,
    "random_state"      : 42,

    # RSI thresholds
    "rsi_oversold"      : 30,
    "rsi_overbought"    : 70,
    "rsi_midzone_low"   : 40,
    "rsi_midzone_high"  : 50,

    # Strategy thresholds
    "volume_spike_mult" : 1.5,
    "tiny_candle_pct"   : 0.3,
    "sr_lookback"       : 20,

    # Indicator settings
    "atr_period"        : 14,
    "supertrend_period" : 10,
    "supertrend_mult"   : 3.0,
}