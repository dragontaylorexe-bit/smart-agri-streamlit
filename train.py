# train.py
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path("data/sensors.csv")
MODEL_PATH = Path("models/model_30s.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

REFRESH_SEC = 5
WINDOW_STEPS = 6           # 30s cửa sổ
HORIZON_STEPS = 6          # dự đoán +30s

# ✅ THỨ TỰ CHUẨN (PHẢI GIỐNG collect.py và app.py)
FEATURES = ["temp", "humidity", "soil", "ph"]

def make_supervised(df: pd.DataFrame, window: int, horizon: int):
    X, y = [], []
    values = df[FEATURES].to_numpy()

    # i là index kết thúc cửa sổ quá khứ
    for i in range(window - 1, len(values) - horizon):
        past_window = values[i - window + 1 : i + 1].reshape(-1)  # (window*4,)
        future = values[i + horizon]                              # (4,)
        X.append(past_window)
        y.append(future)

    return np.array(X), np.array(y)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run collect.py first.")

    df = pd.read_csv(DATA_PATH)

    # basic clean
    df = df.dropna(subset=FEATURES).copy()

    # sort by time if ts exists
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.sort_values("ts").dropna(subset=["ts"])

    X, y = make_supervised(df, WINDOW_STEPS, HORIZON_STEPS)

    if len(X) < 200:
        raise ValueError("Data too small. Collect more samples before training.")

    # split time-based: 80/20
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # model multi-output (ExtraTrees support multioutput natively)
    model = ExtraTreesRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae_each = mean_absolute_error(y_test, pred, multioutput="raw_values")

    print("MAE per target (temp, humidity, soil, ph):", mae_each)

    bundle = {
        "model": model,
        "features": FEATURES,
        "refresh_sec": REFRESH_SEC,
        "window_steps": WINDOW_STEPS,
        "horizon_steps": HORIZON_STEPS,
    }
    joblib.dump(bundle, MODEL_PATH)
    print("Saved:", MODEL_PATH)

if __name__ == "__main__":
    main()
