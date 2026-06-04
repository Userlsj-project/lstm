"""
LSTM (35×8 시퀀스) — 베이스라인과 test MAE 비교

사전: train_baseline.py 실행 권장 (metrics_baseline.json)
실행:
  uv add pandas pymysql python-dotenv scikit-learn tensorflow
  uv run python lstm_train.py
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

from ml_shared import (
    FEATURES,
    SEQ_LEN,
    TARGET,
    inverse_power_kw,
    load_baseline_metrics,
    load_joined,
    make_sequences,
    require_rows,
    split_index,
)


def main() -> None:
    df = load_joined()
    require_rows(df, min_rows=SEQ_LEN + 50)
    print(f"[INFO] join 행 수: {len(df)}, 기간: {df['obs_time'].min()} ~ {df['obs_time'].max()}")

    X, y_scaled, scaler = make_sequences(df, SEQ_LEN)
    cut = split_index(len(X))
    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y_scaled[:cut], y_scaled[cut:]

    model = Sequential(
        [
            LSTM(64, input_shape=(SEQ_LEN, len(FEATURES))),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train,
        y_train,
        epochs=8,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_test_kw = inverse_power_kw(scaler, y_test)
    pred_kw = inverse_power_kw(scaler, pred_scaled)
    mae = mean_absolute_error(y_test_kw, pred_kw)
    rmse = float(np.sqrt(mean_squared_error(y_test_kw, pred_kw)))

    print("[LSTM] 과거 35시간 × 8 feature → 다음 power_kw (scaled 학습, kW 지표)")
    print(f"  test MAE  (kW): {mae:.4f}")
    print(f"  test RMSE (kW): {rmse:.4f}")

    base = load_baseline_metrics()
    if base:
        b_mae = base["mae_kw"]
        print("\n[COMPARE] 같은 DB · 뒤 20% 테스트 구간 (지표: MAE kW)")
        print(f"  Baseline (sklearn): {b_mae:.4f}")
        print(f"  LSTM:               {mae:.4f}")
        if mae < b_mae:
            print("  → LSTM이 더 낮은 MAE (이번 데이터·설정 기준)")
        else:
            print("  → Baseline이 더 낮거나 비슷함 (데이터 적음·epoch 부족일 수 있음)")
    else:
        print("\n[HINT] 비교표를 보려면 먼저: uv run python train_baseline.py")


if __name__ == "__main__":
    main()
