# =============================================================
# FILE: trading_system/phase_c/phase_c_main.py
# PURPOSE: Phase C main runner
#          Command: py -3.11 phase_c_main.py
# =============================================================

import warnings
warnings.filterwarnings("ignore")

from config_c          import CONFIG
from loader_c          import load_phase_b_data
from structure_signals import add_structure_signals
from backtest_runner   import (backtest_all_strategies,
                               plot_equity_curves,
                               plot_drawdown,
                               create_ranking_table)
from model_c           import train_phase_c_model
from reporter_c        import generate_full_report
from saver_c           import save_phase_c_outputs


if __name__ == "__main__":
    print("=" * 58)
    print("  PHASE C — STRUCTURE STRATEGIES + BACKTESTING")
    print("  Building on Phase A + B outputs...")
    print("=" * 58)

    # ── Task 1: Load Phase B data ──
    df, model_b, scaler_b = load_phase_b_data()

    # ── Task 2: Add C1-C8 structure signals ──
    df = add_structure_signals(df, CONFIG)

    # ── Task 4: Backtest all 23 strategies ──
    results = backtest_all_strategies(df, CONFIG)

    # ── Task 5: Equity curve plots ──
    plot_equity_curves(results, df)

    # ── Task 6: Drawdown charts ──
    plot_drawdown(results, df)

    # ── Task 7: Strategy ranking ──
    rank_df = create_ranking_table(results, df)

    # ── Task 8: Train Phase C model ──
    model_c, scaler_c, feat_cols, df = train_phase_c_model(
        df, model_b, scaler_b, rank_df
    )

    # ── Task 9: Full report ──
    top_strategies, avoid_list = generate_full_report(
        results, rank_df, df, CONFIG
    )

    # ── Task 10: Save all outputs ──
    save_phase_c_outputs(
        model_c, scaler_c, df,
        results, top_strategies
    )