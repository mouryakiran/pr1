"""Headless demo: runs the full pipeline and prints an optimization report.

Usage: python demo.py [region] [horizon_hours]

Runs forecasting, anomaly detection, recommendations and impact estimation
without needing the Streamlit UI - handy for quick verification.
"""

import sys
import time

import agent
import energy


def run(region=None, horizon=48):
    started = time.time()
    df = energy.load_data()
    models, stats = energy.load_models()
    print(f"[load] data+models in {time.time()-started:.2f}s\n")

    regions = [region] if region else sorted(df["Region"].unique())
    steps = max(1, horizon // energy.STEP_HOURS)

    for r in regions:
        print("=" * 78)
        print(f"REGION {r}   (test MAPE {stats[r]['test_mape_pct']}%, "
              f"{int(stats[r]['rows']):,} rows)")
        print("=" * 78)

        rdf = df[df["Region"] == r].sort_values("Datetime").reset_index(drop=True)
        adf = energy.detect_anomalies(rdf)

        fc_start = time.time()
        fdf = energy.forecast_region(models[r], rdf, steps)
        print(f"\n-- Forecast ({horizon}h / next {steps} steps) [{time.time()-fc_start:.3f}s]")
        fpeak = fdf["Consumption"].max()
        fpeak_t = fdf.loc[fdf["Consumption"].idxmax(), "Datetime"]
        print(f"   peak {fpeak:,.0f} kW at {fpeak_t:%Y-%m-%d %H:%M}  |  "
              f"avg {fdf['Consumption'].mean():,.0f} kW  |  horizon energy "
              f"{fdf['Consumption'].sum():,.0f} kWh")

        n_anom = int(adf["anomaly"].sum())
        print(f"-- Anomalies: {n_anom} of {len(adf):,} points "
              f"({n_anom/len(adf)*100:.2f}%)")

        rep = agent.agent_recommend(rdf, fdf)
        print("-- Agent observations:")
        for o in rep["observations"]:
            print(f"   * {o}")
        print("-- Recommendations:")
        for i, a in enumerate(rep["actions"], 1):
            est = a["base_kwh"] * a["reduction_pct"]
            print(f"   {i}. [{a['priority']:<6}] {a['title']}  (~{est:,.0f} kWh)")

        res = agent.estimate_impact(rep["actions"], rate=0.12, co2_factor=0.4)
        print(f"-- Estimated impact: {res['total_kwh']:,.0f} kWh saved, "
              f"{res['total_cost']:,.0f} USD saved, {res['total_co2']:,.0f} kg CO2")
        print()

    print(f"completed in {time.time()-started:.2f}s")


if __name__ == "__main__":
    reg = sys.argv[1] if len(sys.argv) > 1 else None
    hor = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    run(reg, hor)