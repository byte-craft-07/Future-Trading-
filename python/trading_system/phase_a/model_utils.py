# =============================================================
# FILE: trading_system/phase_a/model_utils.py
# PURPOSE: Task 6,7,8 — Split, Scale, Train, Evaluate
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from config import CONFIG, PLOTS_DIR


# =============================================================
# TASK 6 — SPLIT & SCALE
# =============================================================

def split_and_scale(df, config: dict):
    """
    Time-series split (no shuffle) + StandardScaler.
    Scaler sirf train data pe fit hota hai.

    Args:
        df    : Labeled DataFrame with all features + Target.
        config: CONFIG dictionary.

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_cols
    """
    print("\n" + "─" * 50)
    print("✂️  DATA SPLIT & SCALING")
    print("─" * 50)

    # ── Features define karo (raw OHLCV aur Target exclude) ──
    exclude      = ["Date", "Open", "High", "Low", "Close",
                    "Adj Close", "Volume", "Target"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df["Target"].values

    print(f"Feature count : {len(feature_cols)}")
    print(f"Features      : {feature_cols}")

    # ── Time-Series Split — NO SHUFFLE ──
    split_idx       = int(len(X) * (1 - config["test_size"]))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nTrain samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")

    # ── Scaling — ONLY fit on train ──
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("✅ Scaling done (fitted on train only — no leakage)")
    return X_train, X_test, y_train, y_test, scaler, feature_cols


# =============================================================
# TASK 7 — TRAIN MODELS
# =============================================================

def train_models(X_train: np.ndarray, y_train: np.ndarray):
    """
    2 baseline models train karo: Logistic Regression + Random Forest.

    Args:
        X_train: Scaled training features.
        y_train: Training labels.

    Returns:
        (lr_model, rf_model) tuple.
    """
    print("\n" + "─" * 50)
    print("🤖 MODEL TRAINING")
    print("─" * 50)

    # ── Model 1: Logistic Regression (baseline) ──
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter     = 1000,
        random_state = CONFIG["random_state"],
        n_jobs       = 1
    )
    lr_model.fit(X_train, y_train)
    print("✅ Logistic Regression training complete.")

    # ── Model 2: Random Forest ──
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators = 100,
        max_depth    = 5,               # Shallow = CPU safe + no overfit
        random_state = CONFIG["random_state"],
        n_jobs       = 1
    )
    rf_model.fit(X_train, y_train)
    print("✅ Random Forest training complete.")

    return lr_model, rf_model


# =============================================================
# TASK 8 — EVALUATE
# =============================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   name: str, feature_cols: list = None) -> dict:
    """
    Model evaluate karo — accuracy, F1, confusion matrix, feature importance.

    Args:
        model       : Trained sklearn model.
        X_test      : Scaled test features.
        y_test      : True labels.
        name        : Model name (filename ke liye).
        feature_cols: Feature names list (RF importance ke liye).

    Returns:
        Dict with model name, accuracy, f1.
    """
    print(f"\n{'═'*50}")
    print(f"📈 EVALUATION: {name}")
    print(f"{'═'*50}")

    y_pred = model.predict(X_test)

    # ── 8a: Accuracy ──
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy      : {acc:.4f} ({acc*100:.2f}%)")

    # ── 8b: Classification Report ──
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["DOWN (0)", "UP (1)"]))

    f1 = f1_score(y_test, y_pred, average="weighted")

    # ── 8c: Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred DOWN", "Pred UP"],
                yticklabels=["True DOWN", "True UP"], ax=ax)
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    cm_path = PLOTS_DIR / f"confusion_matrix_{name.replace(' ', '_')}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix saved: {cm_path}")

    # ── 8d: Feature Importance (RF only) ──
    if "Random Forest" in name and feature_cols is not None:
        importances  = model.feature_importances_
        indices      = np.argsort(importances)[::-1][:10]
        top_features = [feature_cols[i] for i in indices]
        top_values   = importances[indices]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(top_features[::-1], top_values[::-1],
                       color="steelblue", edgecolor="white")
        ax.set_title("Top 10 Feature Importances — Random Forest")
        ax.set_xlabel("Importance Score")
        ax.bar_label(bars, fmt="%.4f", padding=3)
        plt.tight_layout()
        fi_path = PLOTS_DIR / "feature_importance_rf.png"
        plt.savefig(fi_path, dpi=150)
        plt.close()
        print(f"📊 Feature importance saved: {fi_path}")

    return {"model": name, "accuracy": acc, "f1": f1}


def compare_models(results: list):
    """
    Dono models compare karo, winner declare karo.

    Args:
        results: List of dicts from evaluate_model().
    """
    print(f"\n{'═'*50}")
    print("🏆 MODEL COMPARISON")
    print(f"{'═'*50}")
    print(f"{'Model':<25} {'Accuracy':>10} {'F1 Score':>10}")
    print("─" * 50)

    best = max(results, key=lambda r: r["f1"])
    for r in results:
        tag = " ← WINNER" if r["model"] == best["model"] else ""
        print(f"{r['model']:<25} {r['accuracy']:>9.4f} {r['f1']:>9.4f}{tag}")

    print(f"\n✅ {best['model']} wins — F1={best['f1']:.4f}")
    print("   F1 zyada important hai trading mein — class balance")
    print("   aur dono classes ki quality dono dekhta hai.")