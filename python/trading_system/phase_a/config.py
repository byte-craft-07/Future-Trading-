# =============================================================
# FILE: trading_system/phase_a/config.py
# PURPOSE: Central config + all folder paths — ek jagah se
#          sab kuch change hoga, baaki files automatically
#          updated ho jaayengi
# =============================================================

from pathlib import Path

# ── All settings in one place ────────────────────────────────
CONFIG = {
    "ticker"      : "RELIANCE.NS",  # NSE India stock
    "start_date"  : "2018-01-01",
    "end_date"    : "2024-01-01",
    "interval"    : "1d",           # Daily candles
    "test_size"   : 0.2,            # 20% test data
    "random_state": 42,
}

# ── Folder Paths (pathlib — OS safe) ─────────────────────────
BASE_DIR   = Path(__file__).parent          # phase_a/ folder

DATA_RAW   = BASE_DIR / "data" / "raw"
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR  = BASE_DIR / "plots"

# ── Auto-create folders if missing ───────────────────────────
for _folder in [DATA_RAW, DATA_PROC, MODELS_DIR, PLOTS_DIR]:
    _folder.mkdir(parents=True, exist_ok=True)