# =============================================================
# FILE: trading_system/phase_b/model_b.py
# PURPOSE: Task 6,7,8 — Features prepare, train, evaluate,
#          Phase A vs Phase B comparison
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score

from config_b import CONFIG, PHASE_A_FEATURES, PLOTS_DIR_B
from confluence import STRATEGY_COLS


# ── Phase B feature set definition ────────────────────────────
PHASE_B_NEW = [
    "ATR_14", "Supertrend_Signal", "VWAP_Daily",
    "Candle_Body_Ratio", "Bullish_Engulf",
    "Confluence_Score", "Confluence_Strong"
]
PHASE_B_FEATURES = PHASE_A_FEATURES + PHASE_B_NEW + STRATEGY_COLS


# =============================================================
# TASK 6 — PREPARE FEATURES
# =============================================================

def prepare_phase_b_features(df: pd.DataFrame):
    """
    Phase B ka full feature set prepare karo.
    Phase A (14) + New Indicators (7) + Strategies (15) = 36.

    Args:
        df: DataFrame with all features + Target.

    Returns:
        X_train, X_test, y_train, y_test, scaler_b, feature_list
    """
    print("\n" + "─" * 55)
    print("✂️  PHASE B FEATURE PREP + SPLIT")
    print("─" * 55)

    # ── Verify all required columns exist ──
    missing = [c for c in PHASE_B_FEATURES if c not in df.columns]
    if missing:
        print(f"⚠️  Missing features: {missing}")

    available = [c for c in PHASE_B_FEATURES if c in df.columns]

    X = df[available].values
    y = df["Target"].values

    print(f"Phase A features   : {len(PHASE_A_FEATURES)}")
    print(f"New B indicators   : {len(PHASE_B_NEW)}")
    print(f"Strategy signals   : {len(STRATEGY_COLS)}")
    print(f"Total Phase B feat : {len(available)}")

    # ── Time-series split — NO SHUFFLE ──
    split_idx       = int(len(X) * (1 - CONFIG["test_size"]))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nTrain samples      : {len(X_train)}")
    print(f"Test samples       : {len(X_test)}")

    # ── Scale — fit only on train ──
    scaler_b = StandardScaler()
    X_train  = scaler_b.fit_transform(X_train)
    X_test   = scaler_b.transform(X_test)

    print("✅ Scaling complete (no leakage)")

    return X_train, X_test, y_train, y_test, scaler_b, available


# =============================================================
# TASK 7 — TRAIN PHASE B MODELS
# =============================================================

def train_phase_b_models(X_train: np.ndarray, y_train: np.ndarray):
    """
    Phase B ke 2 models train karo:
      - Random Forest (upgraded: deeper + balanced)
      - Gradient Boosting (NEW: sequential learner)

    Args:
        X_train: Scaled training features.
        y_train: Training labels.

    Returns:
        (rf_b, gb_b) — both trained models.
    """
    print("\n" + "─" * 55)
    print("🤖 PHASE B MODEL TRAINING")
    print("─" * 55)

    # ── Model 1: Random Forest (upgraded) ──
    print("Training Phase B Random Forest...")
    rf_b = RandomForestClassifier(
        n_estimators = 100,
        max_depth    = 6,                  # Deeper than Phase A (was 5)
        random_state = CONFIG["random_state"],
        class_weight = "balanced",         # NEW: handles imbalance
        n_jobs       = 1
    )
    rf_b.fit(X_train, y_train)
    print("✅ Random Forest (Phase B) complete.")

    # ── Model 2: Gradient Boosting (NEW) ──
    # Sequential trees — each tree fixes prev errors
    print("\nTraining Gradient Boosting...")
    gb_b = GradientBoostingClassifier(
        n_estimators  = 100,
        max_depth     = 4,
        learning_rate = 0.05,              # Slow = careful learner
        subsample     = 0.8,               # CPU safe (uses 80% rows)
        random_state  = CONFIG["random_state"]
    )
    gb_b.fit(X_train, y_train)
    print("✅ Gradient Boosting complete.")

    return rf_b, gb_b


# =============================================================
# TASK 8 — EVALUATE + COMPARE
# =============================================================

def evaluate_and_compare(model_a, rf_b, gb_b,
                         df: pd.DataFrame,
                         X_test_b: np.ndarray,
                         y_test: np.ndarray,
                         scaler_a, feature_cols_b: list):
    """
    Phase A vs Phase B dono models compare karo.
    Accuracy, F1, Feature Importance chart.

    Args:
        model_a      : Phase A trained model
        rf_b, gb_b   : Phase B models
        df           : Full DataFrame (for Phase A features)
        X_test_b     : Phase B scaled test features
        y_test       : True labels
        scaler_a     : Phase A scaler (for Phase A predictions)
        feature_cols_b: Phase B feature column list

    Returns:
        results list, best_model, best_name
    """
    print("\n" + "─" * 55)
    print("📊 PHASE A vs PHASE B COMPARISON")
    print("─" * 55)

    # ── Prepare Phase A test set (same test rows) ──
    split_idx = int(len(df) * (1 - CONFIG["test_size"]))
    df_test   = df.iloc[split_idx:].reset_index(drop=True)

    # Phase A features only, scaled with Phase A scaler
    X_test_a  = scaler_a.transform(
        df_test[PHASE_A_FEATURES].values
    )

    # ── Predictions from all 3 models ──
    pred_a  = model_a.predict(X_test_a)
    pred_rf = rf_b.predict(X_test_b)
    pred_gb = gb_b.predict(X_test_b)

    # ── Metrics ──
    results = []
    for name, pred in [("Phase A RF",    pred_a),
                       ("Phase B RF",    pred_rf),
                       ("Phase B GB",    pred_gb)]:
        acc = accuracy_score(y_test, pred)
        f1  = f1_score(y_test, pred, average="weighted")
        results.append({"model": name, "acc": acc, "f1": f1})

    # ── Print comparison table ──
    print(f"\n{'Model':<20} {'Accuracy':>10} {'F1 Score':>10}")
    print("─" * 43)
    best = max(results, key=lambda r: r["f1"])
    for r in results:
        tag = " ← BEST" if r["model"] == best["model"] else ""
        print(f"{r['model']:<20} {r['acc']*100:>9.2f}% "
              f"{r['f1']:>9.4f}{tag}")

    # ── Classification report for best Phase B model ──
    best_pred = pred_gb if "GB" in best["model"] else pred_rf
    print(f"\nDetailed Report — {best['model']}:")
    print(classification_report(y_test, best_pred,
                                target_names=["DOWN", "UP"]))

    # ── 8c: Comparison bar chart ──
    _plot_comparison(results)

    # ── 8d: Feature importance (Phase B RF) ──
    _plot_feature_importance(rf_b, feature_cols_b)

    # ── 8e: Which strategies in top 10? ──
    importances  = rf_b.feature_importances_
    top10_idx    = np.argsort(importances)[::-1][:10]
    top10_feats  = [feature_cols_b[i] for i in top10_idx]
    strat_in_top = [f for f in top10_feats
                    if f.startswith("S") and f[1:3].replace("_","").isdigit()]

    print("\n🏆 Strategy signals in TOP 10 features:")
    if strat_in_top:
        for s in strat_in_top:
            print(f"  ✅ {s}")
    else:
        print("  ℹ️  No strategy signals in top 10")

    best_model = gb_b if "GB" in best["model"] else rf_b
    best_name  = best["model"]
    return results, best_model, best_name


def _plot_comparison(results: list):
    """Side-by-side bar chart: Accuracy + F1 for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    names  = [r["model"] for r in results]
    accs   = [r["acc"] * 100 for r in results]
    f1s    = [r["f1"] for r in results]
    colors = ["#aec6cf", "#77dd77", "#fdfd96"]  # Blue, Green, Yellow

    # Accuracy bars
    bars1 = ax1.bar(names, accs, color=colors, edgecolor="gray", width=0.5)
    ax1.set_title("Accuracy Comparison (%)")
    ax1.set_ylabel("Accuracy %")
    ax1.set_ylim(40, 75)
    ax1.bar_label(bars1, fmt="%.2f%%", padding=3)
    ax1.grid(True, axis="y", alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=15, ha="right")

    # F1 bars
    bars2 = ax2.bar(names, f1s, color=colors, edgecolor="gray", width=0.5)
    ax2.set_title("F1 Score Comparison")
    ax2.set_ylabel("Weighted F1 Score")
    ax2.set_ylim(0.3, 0.8)
    ax2.bar_label(bars2, fmt="%.4f", padding=3)
    ax2.grid(True, axis="y", alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")

    plt.suptitle("Phase A vs Phase B Model Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = PLOTS_DIR_B / "phase_a_vs_b_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n📊 Comparison chart saved: {path}")


def _plot_feature_importance(rf_model, feature_cols: list):
    """Top 20 features — strategy signals alag color mein."""
    importances  = rf_model.feature_importances_
    top20_idx    = np.argsort(importances)[::-1][:20]
    top20_feats  = [feature_cols[i] for i in top20_idx]
    top20_vals   = importances[top20_idx]

    # Strategy signal = green, others = steelblue
    colors = [
        "seagreen" if (f.startswith("S") and len(f) > 2
                       and f[1:3].replace("_","").isdigit())
        else "steelblue"
        for f in top20_feats
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top20_feats[::-1], top20_vals[::-1],
                   color=colors[::-1], edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_title("Phase B — Top 20 Feature Importances\n"
                 "(Green = Strategy Signal, Blue = Technical Indicator)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR_B / "feature_importance_b.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Feature importance saved: {path}")

# EXPLANATION: Phase A vs Phase B comparison se pata
# chalega ki strategies add karne ka kya fayda hua.
# Feature importance chart mein green bars = strategy
# signals hain — agar ye blue bars se upar hain to
# strategies genuinely helpful hain model ke liye.