import os

import joblib
import numpy as np
import pandas as pd

DATA_FILE = "processed_energy_data.csv"
MODEL_DIR = "models"
STEP_HOURS = 5

WEEKDAY_CODE = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}
WEEKDAY_NAME = {v: k for k, v in WEEKDAY_CODE.items()}

FEATURES = [
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "Weekday", "Weekend", "Peak_Hour",
    "lag1", "roll3",
]


def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["Datetime"])
    df["Weekday"] = df["Weekday"].map(WEEKDAY_CODE)
    return df.sort_values(["Region", "Datetime"]).reset_index(drop=True)


def _time_features(dt):
    hour = dt.hour
    month = dt.month
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "Weekday": WEEKDAY_CODE[dt.day_name()],
        "Weekend": int(dt.dayofweek >= 5),
        "Peak_Hour": int(17 <= hour <= 21),
    }


def load_models():
    stats = joblib.load(os.path.join(MODEL_DIR, "region_stats.joblib"))
    models = {
        r: joblib.load(os.path.join(MODEL_DIR, f"forecast_{r}.joblib"))
        for r in stats
    }
    return models, stats


def forecast_region(model, df_region, steps):
    df = df_region.sort_values("Datetime")
    last_dt = df["Datetime"].iloc[-1]
    hist = df["Consumption"].dropna().tail(3).tolist()

    preds, dts = [], []
    for i in range(1, steps + 1):
        dt = last_dt + pd.Timedelta(hours=STEP_HOURS * i)
        feats = _time_features(dt)
        feats["lag1"] = hist[-1] if hist else 0.0
        feats["roll3"] = float(np.mean(hist)) if hist else 0.0
        row = pd.DataFrame([{f: feats[f] for f in FEATURES}])
        pred = float(max(model.predict(row)[0], 0.0))
        hist.append(pred)
        if len(hist) > 3:
            hist.pop(0)
        preds.append(pred)
        dts.append(dt)
    return pd.DataFrame({"Datetime": dts, "Consumption": preds})


def detect_anomalies(df_region, window=36, threshold=3.5):
    df = df_region.sort_values("Datetime").reset_index(drop=True)
    med = df["Consumption"].rolling(window, center=True, min_periods=6).median()
    q1 = df["Consumption"].rolling(window, center=True, min_periods=6).quantile(0.25)
    q3 = df["Consumption"].rolling(window, center=True, min_periods=6).quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    upper = med + threshold * (0.75 * iqr)
    lower = med - threshold * (0.75 * iqr)
    df["anomaly"] = (df["Consumption"] > upper) | (df["Consumption"] < lower)
    df["baseline"] = med
    return df