import numpy as np
import pandas as pd

CO2_KG_PER_KWH = 0.40


def analyze_stats(df_region):
    df = df_region.sort_values("Datetime").reset_index(drop=True)
    cons = df["Consumption"]
    total = float(cons.sum())
    peak_kw = float(cons.max())
    peak_time = df.loc[cons.idxmax(), "Datetime"]
    avg = float(cons.mean())

    peak_mask = df["Peak_Hour"] == 1
    overnight_mask = df["Hour"].between(22, 23) | df["Hour"].between(0, 6)
    weekend_mask = df["Weekend"] == 1

    peak_share = float(cons[peak_mask].sum() / cons.sum()) if len(cons) else 0.0
    overnight_share = float(cons[overnight_mask].sum() / cons.sum()) if len(cons) else 0.0
    wd = cons[~weekend_mask]
    we = cons[weekend_mask]
    weekend_ratio = float(we.mean() / wd.mean()) if len(we) and len(wd) and wd.mean() else 0.0

    month = df["Month"]
    cooling = cons[month.between(6, 9)]
    heating = cons[month.isin([11, 12, 1, 2])]
    cooling_total = float(cooling.sum())
    heating_total = float(heating.sum())

    d = detect_anomaly_rate(df)
    anomaly_share = d["anomaly_share"]

    return {
        "rows": int(len(df)),
        "total_kwh": total,
        "avg_kw": avg,
        "peak_kw": peak_kw,
        "peak_time": peak_time,
        "peak_share": peak_share,
        "overnight_share": overnight_share,
        "weekend_ratio": weekend_ratio,
        "cooling_total_kwh": cooling_total,
        "heating_total_kwh": heating_total,
        "anomaly_share": anomaly_share,
    }


def detect_anomaly_rate(df):
    q1 = df["Consumption"].quantile(0.25)
    q3 = df["Consumption"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 3.5 * 0.75 * iqr
    lower = q1 - 3.5 * 0.75 * iqr
    flagged = int(((df["Consumption"] > upper) | (df["Consumption"] < lower)).sum())
    return {"anomaly_share": flagged / len(df) if len(df) else 0.0, "anomaly_count": flagged}


def _action(tid, title, priority, reduction_pct, base_kwh, reasoning):
    return {
        "id": tid,
        "title": title,
        "priority": priority,
        "reduction_pct": reduction_pct,
        "base_kwh": base_kwh,
        "reasoning": reasoning,
    }


def agent_recommend(df_region, forecast_df=None):
    stats = analyze_stats(df_region)
    obs = []
    actions = []

    obs.append(f"Region averaged {stats['avg_kw']:,.0f} kW and consumed "
               f"{stats['total_kwh']:,.0f} kWh in the lookback window.")
    obs.append(f"Peak load was {stats['peak_kw']:,.0f} kW at {stats['peak_time']:%Y-%m-%d %H:%M}.")
    obs.append(f"Peak (17:00-21:00) hours carried {stats['peak_share']*100:.1f}% of all energy.")
    obs.append(f"Weekend consumption is {stats['weekend_ratio']*100:.1f}% of weekday level.")
    obs.append(f"Cooling-season (Jun-Sep) energy: {stats['cooling_total_kwh']:,.0f} kWh; "
               f"heating-season (Nov-Feb): {stats['heating_total_kwh']:,.0f} kWh.")
    obs.append(f"Anomalous readings made up {stats['anomaly_share']*100:.2f}% of the window.")

    if stats["peak_share"] >= 0.16:
        base = stats["total_kwh"] * stats["peak_share"]
        actions.append(_action(
            "shift", "Shift non-critical loads off peak (17:00-21:00)",
            "High", 0.05, base,
            f"Peak-hour share is {stats['peak_share']*100:.1f}% - demand charges spike here. "
            "Shifting washing/drying, pre-cooling and batch processes avoids ~5% of peak energy.",
        ))

    if stats["cooling_total_kwh"] > 0:
        actions.append(_action(
            "hvac_c", "Optimize HVAC cooling setpoints (Jun-Sep)",
            "High", 0.03, stats["cooling_total_kwh"],
            "Cooling drives demand in warm months. Raising setpoints 1-2C during occupied hours "
            "and pre-cooling during off-peak saves ~3% of cooling-period energy.",
        ))

    if stats["heating_total_kwh"] > 0:
        actions.append(_action(
            "hvac_h", "Optimize HVAC heating schedule (Nov-Feb)",
            "Medium", 0.03, stats["heating_total_kwh"],
            "Heating-season energy is significant. Night setback + morning warm-up scheduling "
            "saves ~3% of heating-period energy.",
        ))

    if stats["weekend_ratio"] >= 0.80:
        base = stats["total_kwh"] * (1 - 1 / (1 + stats["weekend_ratio"])) * 0.28
        actions.append(_action(
            "weekend", "Tighten weekend / holiday occupancy schedule",
            "Medium", 0.06, base,
            f"Weekend consumption is {stats['weekend_ratio']*100:.0f}% of weekdays - equipment may "
            "run when spaces are empty. Shut down HVAC/lights outside occupancy saves ~6% of "
            "weekend energy.",
        ))

    if stats["overnight_share"] >= 0.10:
        base = stats["total_kwh"] * stats["overnight_share"]
        actions.append(_action(
            "standby", "Cut standby / phantom loads overnight (22:00-06:00)",
            "Low", 0.02, base,
            f"Overnight loads are {stats['overnight_share']*100:.1f}% of total. Power-management "
            "policies on idle equipment cut ~2% of overnight energy.",
        ))

    if forecast_df is not None and len(forecast_df):
        fpeak = float(forecast_df["Consumption"].max())
        if fpeak > stats["avg_kw"] * 1.15:
            actions.append(_action(
                "dr", "Prepare demand-response event for forecast peak",
                "High", 0.04, fpeak * len(forecast_df),
                f"Forecast peak {fpeak:,.0f} kW is {fpeak/stats['avg_kw']:.0%} above the recent "
                "average - a good candidate for a pre-scheduled demand response event (~4% "
                "reduction on peak windows).",
            ))

    if stats["anomaly_share"] >= 0.01:
        actions.append(_action(
            "anom", "Audit recurring consumption anomalies",
            "Medium", 0.02, stats["total_kwh"],
            f"{stats['anomaly_share']*100:.2f}% of readings deviated strongly from baseline - "
            "these often trace to stuck valves, failed sensors or equipment faults. Fixing them "
            "reduces waste by ~2% of total energy.",
        ))

    actions.sort(key=lambda a: (a["priority"] != "High", -a["base_kwh"]))
    return {"stats": stats, "observations": obs, "actions": actions}


def estimate_impact(actions, rate, co2_factor):
    out = []
    total_kwh = total_cost = total_co2 = 0.0
    for a in actions:
        if not a.get("enabled", True):
            continue
        saved_kwh = a["base_kwh"] * a["reduction_pct"]
        saved_cost = saved_kwh * rate
        co2 = saved_kwh * co2_factor
        total_kwh += saved_kwh
        total_cost += saved_cost
        total_co2 += co2
        out.append({**a, "saved_kwh": saved_kwh, "saved_cost": saved_cost, "co2_kg": co2})
    return {"items": out, "total_kwh": total_kwh, "total_cost": total_cost, "total_co2": total_co2}