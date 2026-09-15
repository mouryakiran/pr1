import os

import numpy as np
import pandas as pd
import pytest

import agent
import energy

REGION = "AEP"


@pytest.fixture(scope="module")
def models():
    if not os.path.exists(os.path.join(energy.MODEL_DIR, "region_stats.joblib")):
        pytest.skip("models not trained - run python train_models.py")
    models, stats = energy.load_models()
    return models, stats


@pytest.fixture(scope="module")
def df():
    return energy.load_data()


@pytest.fixture(scope="module")
def rdf(df):
    return df[df["Region"] == REGION].sort_values("Datetime").reset_index(drop=True)


def _synthetic(window=200, spike_at=100, spike=1.5):
    dts = pd.date_range("2024-01-01", periods=window, freq="5h")
    vals = np.array(100 + 20 * np.sin(2 * np.pi * dts.hour / 24) +
                    5 * np.sin(2 * np.pi * (dts.dayofweek % 7) / 7))
    if spike_at is not None:
        vals[spike_at] *= (1 + spike)
    return pd.DataFrame({
        "Datetime": dts,
        "Consumption": vals,
        "Hour": dts.hour,
        "Month": dts.month,
        "Weekday": [energy.WEEKDAY_CODE[d.day_name()] for d in dts],
        "Weekend": (dts.dayofweek >= 5).astype(int),
        "Peak_Hour": ((dts.hour >= 17) & (dts.hour <= 21)).astype(int),
    })


def test_load_data_schema(df):
    assert {"Datetime", "Consumption", "Region"}.issubset(df.columns)
    assert df["Weekday"].between(0, 6).all()
    assert df["Consumption"].notna().all()
    assert df["Region"].nunique() == 6


def test_forecast_length_and_dates(models, rdf):
    model = models[0][REGION]
    fdf = energy.forecast_region(model, rdf, steps=14)
    assert len(fdf) == 14
    assert (fdf["Consumption"] >= 0).all()
    diff = fdf["Datetime"].diff().dropna().apply(lambda x: x.total_seconds() / 3600)
    assert (diff == energy.STEP_HOURS).all()


def test_forecast_on_synthetic(models):
    model = models[0][REGION]
    syn = energy.load_data()
    syn_df = syn[syn["Region"] == REGION]
    fdf = energy.forecast_region(model, syn_df, steps=5)
    assert len(fdf) == 5
    assert np.isfinite(fdf["Consumption"]).all()


def test_detect_anomalies_flags_spike():
    d = _synthetic()
    out = energy.detect_anomalies(d)
    assert out.loc[out["anomaly"], "Consumption"].max() > \
        out["Consumption"].median() * 1.5
    assert int(out["anomaly"].sum()) >= 1


def test_detect_anomalies_no_flag_on_clean():
    d = _synthetic(spike_at=None)
    out = energy.detect_anomalies(d)
    assert int(out["anomaly"].sum()) == 0


def test_analyze_stats_fields(rdf):
    s = agent.analyze_stats(rdf)
    assert s["total_kwh"] > 0
    assert 0 <= s["peak_share"] <= 1
    assert 0 <= s["weekend_ratio"]
    assert s["peak_kw"] == rdf["Consumption"].max()


def test_agent_recommend_schema(models, rdf):
    model = models[0][REGION]
    fdf = energy.forecast_region(model, rdf, steps=14)
    rep = agent.agent_recommend(rdf, fdf)
    assert len(rep["observations"]) >= 4
    for a in rep["actions"]:
        assert a["id"]
        assert a["priority"] in {"High", "Medium", "Low"}
        assert a["base_kwh"] > 0
        assert 0 < a["reduction_pct"] <= 1
        assert a["reasoning"]


def test_estimate_impact_math():
    actions = [
        {"id": "a", "title": "t", "priority": "High", "reduction_pct": 0.05,
         "base_kwh": 1000, "reasoning": "r", "enabled": True},
        {"id": "b", "title": "t2", "priority": "Low", "reduction_pct": 0.10,
         "base_kwh": 500, "reasoning": "r", "enabled": True},
    ]
    res = agent.estimate_impact(actions, rate=0.12, co2_factor=0.4)
    assert res["total_kwh"] == pytest.approx(50 + 50)
    assert res["total_cost"] == pytest.approx(100 * 0.12)
    assert res["total_co2"] == pytest.approx(100 * 0.4)
    assert len(res["items"]) == 2


def test_estimate_impact_disabled_skipped():
    actions = [
        {"id": "a", "title": "t", "priority": "High", "reduction_pct": 0.05,
         "base_kwh": 1000, "reasoning": "r", "enabled": False},
    ]
    res = agent.estimate_impact(actions, rate=0.12, co2_factor=0.4)
    assert res["total_kwh"] == 0
    assert res["items"] == []