# =============================================================
# FILE: trading_system/phase_e/lstm_models.py
# PURPOSE: Task 3+4 — LSTM Price + Volatility Models (CPU)
# NOTE: TensorFlow CPU-only mode force kiya gaya hai
# =============================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Force CPU

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")   # Double-sure CPU only
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed. "
          "LSTM models will be skipped.")
    print("   Install: py -3.11 -m pip install tensorflow")

from sklearn.preprocessing import MinMaxScaler
from config_e import CONFIG


def create_sequences(data: np.ndarray,
                     seq_len: int) -> tuple:
    """
    Time-series sequences banao for LSTM.
    Input:  seq_len candles
    Output: next candle value

    Args:
        data   : 2D numpy array (n_samples × n_features)
        seq_len: Sequence length (20 candles)

    Returns:
        X (n × seq_len × features), y (n,)
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, 0])   # First col = target
    return np.array(X), np.array(y)


def build_lstm_price_model(df: pd.DataFrame,
                            config: dict):
    """
    Task 3: LSTM Price Predictor
    Architecture: LSTM(72) → Dropout → LSTM(48) → Dropout → Dense(1)
    Target: Close price 2 candles ahead

    Args:
        df    : Intraday DataFrame.
        config: CONFIG dict.

    Returns:
        lstm_price_model, price_scaler OR (None, None) if TF missing
    """
    if not TF_AVAILABLE:
        print("⚠️  Skipping Price LSTM (TF not available)")
        return None, None

    print("\n" + "─" * 55)
    print("🧠 BUILDING LSTM PRICE MODEL")
    print("─" * 55)

    seq_len  = config["sequence_length"]       # 20
    n_ahead  = config["predict_candles"]       # 2

    # ── Select features for price LSTM ──
    price_feats = ["Close", "VWAP", "EMA_9", "EMA_21",
                   "RSI_14", "MACD", "ATR_14",
                   "Volume_MA_20", "Master_Score"]
    avail = [f for f in price_feats if f in df.columns]
    data  = df[avail].ffill().dropna().values

    # ── Scale to [0,1] ──
    price_scaler = MinMaxScaler()
    data_scaled  = price_scaler.fit_transform(data)

    # ── Create sequences ──
    X, y = create_sequences(data_scaled, seq_len)

    # ── Time-series split ──
    split   = int(len(X) * (1 - config["test_size"]))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    print(f"  Features used   : {avail}")
    print(f"  Sequences       : {len(X)} (train={split})")
    print(f"  Input shape     : {Xtr.shape}")

    # ── Build model ──
    units  = config["lstm_price_units"]   # [72, 48]
    drop   = config["dropout_rate"]       # 0.2

    model = Sequential([
        LSTM(units[0], return_sequences=True,
             input_shape=(seq_len, len(avail))),
        Dropout(drop),
        LSTM(units[1], return_sequences=False),
        Dropout(drop),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    print(f"\n  Architecture    : LSTM({units[0]}) → "
          f"Dropout → LSTM({units[1]}) → Dense(1)")
    model.summary(print_fn=lambda x: print(f"  {x}"))

    # ── Train ──
    early_stop = EarlyStopping(monitor="val_loss",
                                patience=3,
                                restore_best_weights=True)
    history = model.fit(
        Xtr, ytr,
        epochs          = config["lstm_epochs"],
        batch_size      = config["lstm_batch"],
        validation_split= config["validation_split"],
        callbacks       = [early_stop],
        verbose         = 0
    )

    # ── Evaluate ──
    pred_scaled = model.predict(Xte, verbose=0).flatten()

    # Inverse transform (reconstruct full feature array for inverse)
    dummy = np.zeros((len(pred_scaled), len(avail)))
    dummy[:, 0] = pred_scaled
    pred_actual = price_scaler.inverse_transform(dummy)[:, 0]

    dummy_true = np.zeros((len(yte), len(avail)))
    dummy_true[:, 0] = yte
    true_actual = price_scaler.inverse_transform(dummy_true)[:, 0]

    mae  = np.mean(np.abs(true_actual - pred_actual))
    mape = np.mean(np.abs((true_actual - pred_actual)
                           / (true_actual + 1e-8))) * 100

    print(f"\n  ✅ Price LSTM trained   (epochs={len(history.epoch)})")
    print(f"     MAE  : ₹{mae:.2f}")
    print(f"     MAPE : {mape:.2f}%")

    return model, price_scaler


def build_lstm_volatility_model(df: pd.DataFrame,
                                 config: dict):
    """
    Task 4: LSTM Volatility Predictor
    Architecture: LSTM(50) → Dropout → LSTM(30) → Dropout → Dense(1)
    Target: ATR (volatility) 2 candles ahead

    Args:
        df    : Intraday DataFrame.
        config: CONFIG dict.

    Returns:
        lstm_vol_model, vol_scaler OR (None, None) if TF missing
    """
    if not TF_AVAILABLE:
        print("⚠️  Skipping Volatility LSTM (TF not available)")
        return None, None

    print("\n" + "─" * 55)
    print("📊 BUILDING LSTM VOLATILITY MODEL")
    print("─" * 55)

    seq_len = config["sequence_length"]

    # ── Volatility features ──
    vol_feats = ["ATR_14", "BB_Upper", "BB_Lower",
                 "High_Low_Range" if "High_Low_Range" in df.columns
                 else "ATR_14",
                 "Volume_MA_20", "RSI_14"]
    avail = list(dict.fromkeys(
        [f for f in vol_feats if f in df.columns]
    ))
    data  = df[avail].ffill().dropna().values

    vol_scaler  = MinMaxScaler()
    data_scaled = vol_scaler.fit_transform(data)

    X, y   = create_sequences(data_scaled, seq_len)
    split  = int(len(X) * (1 - config["test_size"]))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    print(f"  Features used   : {avail}")
    units  = config["lstm_vol_units"]      # [50, 30]
    drop   = config["dropout_rate"]

    model = Sequential([
        LSTM(units[0], return_sequences=True,
             input_shape=(seq_len, len(avail))),
        Dropout(drop),
        LSTM(units[1], return_sequences=False),
        Dropout(drop),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss",
                                patience=3,
                                restore_best_weights=True)
    history = model.fit(
        Xtr, ytr,
        epochs           = config["lstm_epochs"],
        batch_size       = config["lstm_batch"],
        validation_split = config["validation_split"],
        callbacks        = [early_stop],
        verbose          = 0
    )

    pred = model.predict(Xte, verbose=0).flatten()
    mae  = np.mean(np.abs(yte - pred))
    print(f"\n  ✅ Volatility LSTM trained (epochs={len(history.epoch)})")
    print(f"     MAE (scaled): {mae:.4f}")

    return model, vol_scaler


def lstm_predict_latest(model, scaler,
                         df: pd.DataFrame,
                         feature_cols: list,
                         seq_len: int) -> float:
    """
    Latest seq_len candles se ek prediction karo.

    Args:
        model       : Trained LSTM model.
        scaler      : Fitted MinMaxScaler.
        df          : Recent DataFrame.
        feature_cols: Feature column names used in training.
        seq_len     : Sequence length.

    Returns:
        Predicted value (inverse-scaled) as float.
    """
    if model is None:
        return 0.0

    avail = [f for f in feature_cols if f in df.columns]
    data  = df[avail].tail(seq_len).ffill().values

    if len(data) < seq_len:
        return 0.0

    try:
        scaled = scaler.transform(data)
        X      = scaled.reshape(1, seq_len, len(avail))
        pred_s = model.predict(X, verbose=0)[0][0]

        # Inverse scale
        dummy      = np.zeros((1, len(avail)))
        dummy[0,0] = pred_s
        inv        = scaler.inverse_transform(dummy)[0][0]
        return float(inv)
    except Exception as e:
        print(f"⚠️  LSTM predict error: {e}")
        return 0.0

# EXPLANATION: LSTM time-series ka sequence dekhta
# hai — pichle 20 candles se next candle predict.
# Price LSTM: actual ₹ value predict karta hai.
# Volatility LSTM: kitna move hoga predict karta hai.
# CPU pe chalane ke liye units chhote rakhe hain
# aur epochs sirf 20 hain.