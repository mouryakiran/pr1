import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

import energy
import agent

st.set_page_config(page_title="Energy Optimization Agent", page_icon="bolt", layout="wide")


@st.cache_data(show_spinner=False)
def cached_data():
    return energy.load_data()


@st.cache_resource(show_spinner=False)
def cached_models():
    return energy.load_models()


@st.cache_data(show_spinner=False)
def region_frame(df, region):
    return df[df["Region"] == region].sort_values("Datetime").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cached_forecast(_model, df_region, steps):
    return energy.forecast_region(_model, df_region, steps)


@st.cache_data(show_spinner=False)
def cached_anomalies(df_region, window, threshold):
    return energy.detect_anomalies(df_region, window, threshold)


def main():
    df = cached_data()
    models, stats = cached_models()

    st.sidebar.title("Energy Agent")
    page = st.sidebar.radio(
        "Navigate",
        ("Dashboard", "Forecast", "Anomaly Detection", "Recommendations", "Impact Estimation"),
    )
    region = st.sidebar.selectbox("Region", sorted(df["Region"].unique()))
    rate = st.sidebar.slider("Electricity rate ($/kWh)", 0.05, 0.30, 0.12, 0.005)
    co2f = st.sidebar.slider("CO2 factor (kg/kWh)", 0.1, 0.9, 0.4, 0.05)

    rdf = region_frame(df, region)
    model = models[region]
    mape = stats[region]["test_mape_pct"]

    if page == "Dashboard":
        page_dashboard(rdf, region, mape, stats)
    elif page == "Forecast":
        page_forecast(rdf, model, region, mape)
    elif page == "Anomaly Detection":
        page_anomalies(rdf, region)
    elif page == "Recommendations":
        page_recommendations(rdf, model, region)
    else:
        page_impact(rdf, model, region, rate, co2f)


def page_dashboard(rdf, region, mape, stats):
    st.header(f"Dashboard - {region}")
    a = agent.analyze_stats(rdf)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total energy", f"{a['total_kwh']:,.0f} kWh")
    c2.metric("Avg load", f"{a['avg_kw']:,.0f} kW")
    c3.metric("Peak load", f"{a['peak_kw']:,.0f} kW")
    c4.metric("Peak-hour share", f"{a['peak_share']*100:.1f}%")
    c5.metric("Anomaly rate", f"{a['anomaly_share']*100:.2f}%")

    st.line_chart(rdf.tail(700).set_index("Datetime")["Consumption"], height=320)

    rep = agent.agent_recommend(rdf)
    top = rep["actions"][0] if rep["actions"] else {"title": "None", "reasoning": ""}
    st.info(f"Top recommendation: {top['title']} - {top.get('reasoning', '')}")
    st.caption(f"Model test MAPE for {region}: {mape}%  |  rows: {int(stats[region]['rows']):,}")


def page_forecast(rdf, model, region, mape):
    st.header(f"Energy Forecast - {region}")
    horizon = st.select_slider("Forecast horizon", options=[24, 48, 72, 120], value=48,
                               help="Hours ahead, in 5-hour steps")
    steps = horizon // energy.STEP_HOURS
    fdf = cached_forecast(model, rdf, steps)
    hist = rdf.tail(400)

    fpeak = float(fdf["Consumption"].max())
    fpeak_time = fdf.loc[fdf["Consumption"].idxmax(), "Datetime"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast peak", f"{fpeak:,.1f} kW")
    c2.metric("Peak at", f"{fpeak_time:%a %d %b, %H:%M}")
    c3.metric("Avg forecast", f"{fdf['Consumption'].mean():,.1f} kW")
    c4.metric("Horizon energy", f"{fdf['Consumption'].sum():,.0f} kWh")

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(hist["Datetime"], hist["Consumption"], lw=1.2, color="#4c78a8", label="History")
    ax.plot(fdf["Datetime"], fdf["Consumption"], lw=1.6, color="#f58518",
            linestyle="--", label="Forecast")
    ax.scatter([fpeak_time], [fpeak], color="#e15759", zorder=5,
               label=f"Peak {fpeak:,.0f} kW")
    ax.axhline(rdf["Consumption"].mean(), color="#9ecae9", ls=":", lw=1, label="Hist. mean")
    ax.set_ylabel("kW")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"{region} - load forecast next {horizon}h (test MAPE {mape}%)")
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    table = fdf.copy()
    table["Time"] = table["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(table[["Time", "Consumption"]].rename(
        columns={"Consumption": "Predicted kW"}), hide_index=True)

    st.session_state["_fdf"] = fdf


def page_anomalies(rdf, region):
    st.header(f"Anomaly Detection - {region}")
    threshold = st.slider("Detection threshold", 2.0, 6.0, 3.5, 0.1)
    adf = cached_anomalies(rdf, 36, threshold)
    flagged = adf[adf["anomaly"]]
    n = int(len(flagged))

    c1, c2, c3 = st.columns(3)
    c1.metric("Anomalies flagged", n)
    c2.metric("Anomaly rate", f"{n/len(adf)*100:.2f}%")
    ratio = (adf["Consumption"] / adf["baseline"]).dropna()
    c3.metric("Max spike vs baseline", f"{ratio.max()-1:+.1%}")

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(adf["Datetime"], adf["Consumption"], lw=1.1, color="#4c78a8", label="Consumption")
    ax.plot(adf["Datetime"], adf["baseline"], lw=1.0, color="#9ecae9", ls="--",
            label="Baseline (rolling median)")
    if n:
        ax.scatter(flagged["Datetime"], flagged["Consumption"], color="#e15759",
                   s=26, zorder=5, label=f"{n} anomalies")
    ax.set_ylabel("kW")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"{region} - anomalies (threshold {threshold})")
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    if n:
        show = flagged[["Datetime", "Consumption", "baseline"]].copy()
        show["Deviation %"] = (show["Consumption"] / show["baseline"] - 1) * 100
        show["Time"] = show["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            show[["Time", "Consumption", "baseline", "Deviation %"]]
            .sort_values("Deviation %", ascending=False)
            .rename(columns={"Consumption": "kW", "baseline": "Baseline kW"}),
            hide_index=True)
        st.caption("Points beyond threshold x (0.75 x IQR) of the rolling median are "
                   "flagged - typical causes: equipment faults, lost sensors, or special events.")
    else:
        st.success("No anomalies detected in the window.")


def page_recommendations(rdf, model, region):
    st.header(f"Agent Recommendations - {region}")
    fdf = st.session_state.get("_fdf")
    if fdf is None or len(fdf) != 14:
        fdf = cached_forecast(model, rdf, 14)
    rep = agent.agent_recommend(rdf, fdf)

    with st.expander("Reasoning trace (what the agent observed)", expanded=False):
        for o in rep["observations"]:
            st.markdown(f"- {o}")

    if not rep["actions"]:
        st.success("Region already looks well-optimized - no strong action triggers found.")
        return

    for i, act in enumerate(rep["actions"]):
        est = act["base_kwh"] * act["reduction_pct"]
        tag = {"High": "HIGH", "Medium": "MED", "Low": "LOW"}.get(act["priority"], "")
        with st.container(border=True):
            st.markdown(f"**{i+1}. [{tag}] {act['title']}**")
            st.caption(act["reasoning"])
            st.markdown(f"Estimated saving: **{est:,.0f} kWh** "
                        f"({est*0.12:,.0f} USD @ 0.12 USD/kWh)")


def page_impact(rdf, model, region, rate, co2f):
    st.header(f"Impact Estimation - {region}")
    fdf = st.session_state.get("_fdf")
    if fdf is None or len(fdf) != 14:
        fdf = cached_forecast(model, rdf, 14)
    rep = agent.agent_recommend(rdf, fdf)
    actions = rep["actions"]

    if not actions:
        st.success("No actions to estimate - no strong action triggers found.")
        return

    st.markdown("**Tune each action and see its impact.**")
    for a in actions:
        with st.container(border=True):
            c1, c2 = st.columns([3, 4])
            a["enabled"] = c1.checkbox("Enable", value=True,
                                       key=f"en_{a['id']}_{region}",
                                       label_visibility="collapsed")
            c1.markdown(f"**{a['title']}**")
            a["reduction_pct"] = c2.slider(
                "Reduction %", 1, 15, int(a["reduction_pct"] * 100),
                key=f"pct_{a['id']}_{region}", label_visibility="collapsed") / 100.0
            c2.caption(a["reasoning"])

    result = agent.estimate_impact(actions, rate, co2f)
    k1, k2, k3 = st.columns(3)
    k1.metric("Total energy saved", f"{result['total_kwh']:,.0f} kWh")
    k2.metric("Cost saved", f"{result['total_cost']:,.0f} USD")
    k3.metric("CO2 avoided", f"{result['total_co2']:,.0f} kg")

    if result["items"]:
        rows = []
        for it in result["items"]:
            rows.append({
                "Action": it["title"],
                "Base (kWh)": it["base_kwh"],
                "Reduction %": it["reduction_pct"],
                "Saved kWh": it["saved_kwh"],
                "Saved USD": it["saved_cost"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.bar([r["Action"][:28] for r in rows], [r["Saved USD"] for r in rows],
               color="#4c78a8")
        ax.set_ylabel("USD saved")
        ax.set_title(f"Estimated savings by action at {rate:.3f} USD/kWh")
        ax.grid(axis="y", alpha=0.25)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
        st.pyplot(fig)


if __name__ == "__main__":
    main()