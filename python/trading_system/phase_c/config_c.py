# =============================================================
# FILE: trading_system/phase_c/config_c.py
# PURPOSE: Phase C CONFIG + all folder paths
# =============================================================

from pathlib import Path

# ── Folder Paths ─────────────────────────────────────────────
BASE_DIR_C  = Path(__file__).parent
PHASE_B_DIR = BASE_DIR_C.parent / "phase_b"

DATA_PROC_C    = BASE_DIR_C / "data" / "processed"
MODELS_DIR_C   = BASE_DIR_C / "models"
PLOTS_DIR_C    = BASE_DIR_C / "plots"
BT_RESULTS_DIR = BASE_DIR_C / "backtest" / "results"
BT_PLOTS_DIR   = BASE_DIR_C / "backtest" / "plots"

# Auto-create all folders
for _f in [DATA_PROC_C, MODELS_DIR_C, PLOTS_DIR_C,
           BT_RESULTS_DIR, BT_PLOTS_DIR]:
    _f.mkdir(parents=True, exist_ok=True)

# ── Phase B Input Paths ───────────────────────────────────────
PHASE_B_CSV     = PHASE_B_DIR / "data" / "processed" / "RELIANCE_phase_b.csv"
PHASE_B_MODEL   = PHASE_B_DIR / "models" / "best_model_phase_b.pkl"
PHASE_B_SCALER  = PHASE_B_DIR / "models" / "scaler_phase_b.pkl"
PHASE_B_STRAT_J = PHASE_B_DIR / "models" / "strategy_columns.json"

# ── All 15 Phase B strategy column names ─────────────────────
PHASE_B_STRATEGIES = [
    "S1_RSI_MACD_Reversal", "S2_SMA_Crossover",
    "S3_RSI_Pullback",      "S4_MACD_Momentum",
    "S5_Volume_Breakout",   "S6_Engulf_MACD",
    "S7_Oversold_Bounce",   "S8_Trend_Follow",
    "S9_Momentum_Breakout", "S10_EMA_Cross",
    "S11_Gap_Up",           "S12_Support_Bounce",
    "S13_Pullback_Entry",   "S14_Tiny_Candle",
    "S15_Supertrend"
]

# ── 8 Phase C structure signal names ─────────────────────────
PHASE_C_STRATEGIES = [
    "C1_False_Breakout",  "C2_False_Trendline",
    "C3_BOS",             "C4_First_Pullback",
    "C5_Retest",          "C6_Range_Expansion",
    "C7_RSI_Divergence",  "C8_Structure_Shift"
]

ALL_STRATEGIES = PHASE_B_STRATEGIES + PHASE_C_STRATEGIES

# ── Settings ─────────────────────────────────────────────────
CONFIG = {
    "test_size"         : 0.2,
    "random_state"      : 42,

    # Structure signal settings
    "sr_period"         : 20,
    "atr_mult"          : 1.5,
    "bos_swing_period"  : 10,
    "range_tight_pct"   : 0.5,

    # Backtest settings
    "initial_capital"   : 100_000,   # ₹1 lakh
    "commission_pct"    : 0.001,     # 0.1% per side
    "slippage_pct"      : 0.0005,    # 0.05% per side
    "position_size"     : 0.1,       # 10% of capital per trade
    "stop_loss_atr"     : 2.0,       # SL = Entry - 2*ATR
    "take_profit_atr"   : 4.0,       # TP = Entry + 4*ATR
    "min_confluence"    : 4,         # Strong signal threshold
}