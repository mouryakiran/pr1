# Energy Optimization Agent

Buildings, campuses and factories use this agent to **forecast peak energy consumption**,
**detect energy-consumption anomalies**, **recommend energy-saving actions** and
**estimate the monetary/environmental impact** of those actions.

Built around PJM hourly electricity consumption (2015-2024), it runs as a fast,
single-page Streamlit app with five views. Model `MAPE` is ~3-4%.

## Features

| Module | What it does |
|---|---|
| Dashboard | KPIs for the selected region (total, avg/peak load, peak-hour share, anomaly rate) + usage trend. |
| Forecast | Gradient-boosting model per region predicts the next 24-120 h; highlights the forecast peak and its time. |
| Anomaly Detection | Rolling-median IQR detector flags deviations from the baseline; adjustable threshold. |
| Recommendations | A rule-based agent reasons over observed statistics and emits prioritized actions with a reasoning trace. |
| Impact Estimation | Toggle actions and adjust reduction % to estimate kWh, USD and CO2 savings at a configurable rate. |

## Project layout

```
app.py            Streamlit UI (all five views)
energy.py         data loading, feature engineering, forecast + anomaly functions
agent.py          the reasoning agent: statistics, recommendations, impact math
train_models.py   offline training -> models/forecast_<region>.joblib
demo.py           headless end-to-end run of the whole pipeline (no UI)
test_energy.py    pytest suite for data, forecast, anomalies, agent, impact
models/           saved artifacts + region stats
processed_energy_data.csv   6 regions, ~105k rows (fast, used by the app)
```

## Run it

```bash
pip install -r requirements.txt
python train_models.py     # optional: retrain models (only needed once)
python -m streamlit run app.py
```

The app opens at http://localhost:8501. Data loads in ~0.2 s and models in ~1 s
(cached), and every page runs in well under half a second.

## Verify without the UI

```bash
python demo.py               # full pipeline report for all regions
python demo.py AEP 72        # single region, 72h horizon
python -m pytest test_energy.py -q    # 9 unit tests
```

## How it works (brief)

- **Forecast**: features (hour/month cycles, weekday, weekend, peak-hour flag, lag and
  rolling mean) feed a `HistGradientBoostingRegressor` per region, trained on 90% of
  history and validated on the last 10%. Forecasting is recursive over 5-hour steps.
- **Anomalies**: a point is flagged when it lies more than `threshold` x (0.75 x IQR)
  beyond the rolling median of the surrounding ~7.5 day window.
- **Agent**: extracts interpretable statistics (peak share, weekend ratio, cooling/heating
  energy, overnight share, anomaly rate, forecast peak) and triggers domain actions. Each
  action carries a human-readable reasoning note.
- **Impact**: `saved = base_kwh x reduction%`, monetized at a user-set $/kWh and CO2
  factor (default 0.4 kg/kWh).