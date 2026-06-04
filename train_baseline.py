"""
sklearn 베이스라인 (LSTM 없이 발전량 예측)

- 같은 join · 같은 8 feature · 같은 시간순 train/test (뒤 20%)
- 8교시 LSTM 결과와 test MAE(kW) 비교용 metrics 저장

실행:
  uv add pandas pymysql python-dotenv scikit-learn
  uv run python train_baseline.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_shared import (
    FEATURES,
    METRICS_PATH,
    TARGET,
    load_joined,
    require_rows,
    save_baseline_metrics,
    split_index,
)


def main() -> None:
    df = load_joined()
    require_rows(df)
    print(f"[INFO] join 행 수: {len(df)}, 기간: {df['obs_time'].min()} ~ {df['obs_time'].max()}")

    X = df[FEATURES].astype(float).values
    y = df[TARGET].astype(float).values
    cut = split_index(len(df))
    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]

    model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)
    save_baseline_metrics(mae, rmse, r2)

    print("[BASELINE] HistGradientBoosting — 현재 시각 8 feature → power_kw")
    print(f"  test MAE  (kW): {mae:.4f}")
    print(f"  test RMSE (kW): {rmse:.4f}")
    print(f"  test R²:        {r2:.4f}")
    print(f"  (저장) {METRICS_PATH}")
    print("[NEXT] 8교시: uv run python lstm_train.py")


if __name__ == "__main__":
    main()
