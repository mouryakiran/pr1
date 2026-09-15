import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_FILE = "processed_energy_data.csv"
MODEL_DIR = "models"
STEP_HOURS = 5

WEEKDAY_CODE = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}

FEATURES = [
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "Weekday", "Weekend", "Peak_Hour",
    "lag1", "roll3",
]


def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["Datetime"])
    return df.sort_values(["Region", "Datetime"]).reset_index(drop=True)


def add_features(df):
    hour = df["Hour"].values
    month = df["Month"].values
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    df["lag1"] = df["Consumption"].shift(1)
    df["roll3"] = df["Consumption"].rolling(3).mean().shift(1)
    df["Weekday"] = df["Weekday"].map(WEEKDAY_CODE)
    return df


def train_region(df, region):
    d = add_features(df).dropna().reset_index(drop=True)
    split = int(len(d) * 0.9)
    train = d.iloc[:split]
    test = d.iloc[split:]
    model = HistGradientBoostingRegressor(
        max_iter=350, learning_rate=0.06, max_depth=7, random_state=42,
    )
    model.fit(train[FEATURES], train["Consumption"])
    mape_test = np.nan
    if len(test) > 5:
        pred = model.predict(test[FEATURES])
        denom = test["Consumption"].values
        mape_test = float(np.mean(np.abs(pred - denom) / np.maximum(denom, 1)) * 100)
    return model, round(mape_test, 2)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.time()
    df = load_data()
    print(f"loaded {len(df):,} rows in {time.time()-t0:.2f}s")

    stats = {}
    for region in sorted(df["Region"].unique()):
        t = time.time()
        sub = df[df["Region"] == region]
        model, mape = train_region(sub, region)
        joblib.dump(model, os.path.join(MODEL_DIR, f"forecast_{region}.joblib"))
        stats[region] = {
            "rows": int(len(sub)),
            "test_mape_pct": mape,
            "train_time_s": round(time.time() - t, 2),
            "last_datetime": int(sub["Datetime"].max().timestamp()),
            "last_consumption": float(sub["Consumption"].iloc[-1]),
        }
        print(f"  {region}: trained in {stats[region]['train_time_s']}s, test MAPE={mape}%")

    joblib.dump(stats, os.path.join(MODEL_DIR, "region_stats.joblib"))
    print(f"\nAll models saved to '{MODEL_DIR}' in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()