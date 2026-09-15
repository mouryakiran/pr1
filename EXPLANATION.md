
# Project Explanation — Energy Optimization Agent

This file explains **what we built, why, and how it works** — end to end. It is a
walkthrough of the project for anyone (including a reviewer) who wants to
understand it without reading all the code.

---

## 1. The problem we were asked to solve

> Buildings, campuses and factories want to **reduce energy usage and cut costs**
> without disrupting operations. Energy demand changes with time, occupancy,
> weather and usage patterns.
>
> We built an **energy optimization agent** that:
> 1. Forecasts peak energy consumption
> 2. Detects energy-consumption anomalies
> 3. Recommends energy-saving actions
> 4. Estimates the impact of those actions

So the project combines **forecasting**, **optimization** and **agentic AI**
(a small "agent" that observes data, reasons, and makes recommendations).

---

## 2. What we built (in one sentence)

A fast, 5-page **Streamlit app** where you pick a utility region and get: a
dashboard, a load forecast with the next peak highlighted, an anomaly scan, a
list of ranked energy-saving recommendations (with reasoning), and an impact
calculator that converts savings into kWh, dollars and CO₂.

Everything runs on real utility data (PJM hourly consumption, 2015–2024).

---

## 3. The data we used

`processed_energy_data.csv` (the small, fast file):

| Item | Value |
|---|---|
| Rows | 105,207 |
| Regions | 6 (AEP, DAYTON, DOM, EKPC, NI, PJMW) |
| Time span | 2015-01-01 → 2024-12-31 (10 years) |
| Interval | Every 5 hours per region |
| Columns | Datetime, Consumption, Region, Year, Month, Day, Hour, Weekday, Weekend, Peak_Hour |

`Peak_Hour` is `True` for 17:00–21:00 — the hours when utility demand charges are highest.

We deliberately used the small file: the full file is 1M rows and would slow the
app down for no benefit at this "basic level" stage.

---

## 4. Project structure

| File | Purpose |
|---|---|
| `app.py` | The Streamlit user interface (5 pages) |
| `energy.py` | Core engine: loads data, engineers features, runs forecasts and anomaly detection |
| `agent.py` | The "agent": analyzes statistics, generates recommendations with reasoning, computes impact |
| `train_models.py` | Offline script that trains and saves a forecast model per region |
| `demo.py` | Headless version of the whole pipeline (no UI) — great for verification |
| `test_energy.py` | 9 automated unit tests |
| `models/` | Saved trained models + region stats (created by `train_models.py`) |
| `requirements.txt` | Python dependencies |
| `README.md` | Quick run instructions |
| `EXPLANATION.md` | This file |

---

## 5. How each required feature works

### 5.1 Forecasting peak energy consumption

**Model:** one `HistGradientBoostingRegressor` (a fast gradient-boosting tree
model from scikit-learn) trained separately for each of the 6 regions.

**Why this model:** it is non-linear (good for load patterns), trains in under a
second, and works with mixed features without scaling.

**Features used to predict consumption at any given hour:**
- Hour of day, expressed as `sin`/`cos` (models day/night rhythm)
- Month of year, as `sin`/`cos` (models seasons)
- Weekday code (0–6), Weekend flag, Peak_Hour flag
- Lag of previous reading (`lag1`) and 3-reading rolling mean (`roll3`) — captures recent trend

**Training:** first 90% of history trains the model, last 10% is held out to
score it → **measured test error (MAPE): 3–4%** across regions.

**Forecasting forward:** the model predicts the next few time-steps *recursively*
— each new prediction becomes the "previous reading" for the next step, so the
model rolls itself forward 5 hours at a time.

**Verified output (AEP, 72h forecast):**
```
peak 20,662 kW at 2025-01-01 18:00  |  avg 17,842 kW  |  horizon energy 249,781 kWh
```
The app shows history + forecast on one chart and highlights the forecast peak
and its time, which is exactly the "peak energy consumption" requirement.

### 5.2 Detecting energy-consumption anomalies

**Method:** a robust statistical detector per point:
- Compute the **rolling median** around each reading (a ~7.5-day window) = expected "baseline".
- Compute the surrounding **IQR** (interquartile range) to measure normal variability.
- Flag a point as an **anomaly** when it is more than `threshold × (0.75 × IQR)` above or below that baseline.

This is a "trimming-median / IQR" approach (robust to outliers by design).

The app shows the baseline, the actual consumption, and red dots for flagged
anomalies. The threshold is adjustable (2.0–6.0).

**Verified output (AEP):** 186 of 17,535 points (1.06%) flagged; the largest
spikes are listed in a table with % deviation.

**Why it matters:** anomalies often mean equipment faults, stuck sensors, or
special events — catching them saves energy and avoids surprises.

### 5.3 Recommending energy-saving actions (the "agent")

The agent first gathers **observations** from the data (these are shown to the
user as a *reasoning trace*):

- Average load and total energy in the window
- Where the peak load occurred
- What share of energy is consumed during peak hours (17:00–21:00)
- Weekend vs weekday consumption ratio
- Cooling-season (Jun–Sep) and heating-season (Nov–Feb) energy
- How many readings were anomalous

Then it applies simple **rules** — if a pattern is present, it emits an action:

| Trigger observed | Action recommended | Default saving |
|---|---|---|
| Peak-hour share ≥ 16% | Shift non-critical loads off peak | 5% of peak-hour energy |
| Cooling-season energy > 0 | Optimize HVAC cooling setpoints | 3% of cooling energy |
| Heating-season energy > 0 | Optimize HVAC heating schedule | 3% of heating energy |
| Weekend ≈ weekday (>80%) | Tighten weekend/holiday schedule | 6% of weekend energy |
| Overnight loads large (≥10%) | Cut standby/phantom loads | 2% of overnight energy |
| Forecast peak well above average | Prepare demand-response event | 4% on peak windows |
| Many anomalies | Audit recurring anomalies | 2% of total energy |

Every action carries: a **title**, a **priority** (High/Med/Low), an **estimated
kWh saving**, and a **plain-English reasoning note** explaining *why* — that is
the "agentic" feel: it doesn't just list tips, it reasons from your data.

**Verified output (AEP)** — 5 actions generated, top ones:
1. Shift non-critical loads off peak (~3,730,649 kWh)
2. Optimize HVAC cooling setpoints (~3,160,433 kWh)
3. Optimize HVAC heating schedule (~3,106,576 kWh)

### 5.4 Estimating the impact of those actions

**Formula:** `saved kWh = applicable base energy × reduction %`, then
`cost saved = saved kWh × electricity rate`, and
`CO₂ avoided = saved kWh × CO₂ factor`.

In the **Impact Estimation** page you can:
- Enable/disable any action,
- Adjust its reduction percentage with a slider,
- Change the electricity rate ($/kWh) and CO₂ factor in the sidebar.

The app recomputes everything instantly and shows:
- Total kWh, dollars and CO₂ saved,
- A per-action table,
- A bar chart of savings by action.

**Verified output (AEP):** all 5 actions → **14,645,415 kWh / $1,757,450 / 5,858,166 kg CO₂**.

Enabling you to pick the best combination of actions = the **optimization** part.

---

## 6. Data flow (how the pieces connect)

```
processed_energy_data.csv
        │  energy.load_data()                  (~0.2 s, cached)
        ▼
   full dataframe (6 regions)
        │  filter by selected region
        ▼
   one region's series
        ├──► energy.forecast_region()  ──►  next-N-hours forecast + peak  (Forecast page)
        ├──► energy.detect_anomalies() ──►  flagged points + baseline       (Anomaly page)
        └──► agent.analyze_stats() ▸ agent_recommend() ─► ranked actions    (Recommendations)
                                        │
                                        └─► agent.estimate_impact() ─► kWh/$/CO₂  (Impact page)
```

Models are trained once offline (`train_models.py`) and saved to `models/`, so
the app never retrains — that is what keeps the load time short.

---

## 7. Speed (the "loading should not take long" requirement)

Measured on this machine:

| Step | Time |
|---|---|
| Load data | ~0.2 s |
| Load models (first time) | ~1 s |
| Every later feature/page (cached) | < 0.5 s |
| 72h forecast | ~0.1 s |
| Anomaly scan | ~0.04 s |
| Agent reasoning | ~0.01 s |
| Impact estimate | instant |
| Full headless demo, all 6 regions | ~3 s |

Streamlit `@st.cache_data` / `@st.cache_resource` means every page is computed
once and reused — switching between features is effectively instant.

---

## 8. How to run it

```bash
pip install -r requirements.txt
python train_models.py                 # only needed once (models already saved)
python -m streamlit run app.py         # browser UI at http://localhost:8501
python demo.py                         # headless report (no UI)
python -m pytest test_energy.py -q     # 9 unit tests -> all pass
```

---

## 9. Verification summary

- 6 models trained, per-region test error (MAPE) **3.16 – 3.89%**
- Anomaly detector flags real spikes and **passes synthetic spike tests**
- Agent produces realistic, prioritized recommendations with reasoning
- Impact math verified by **unit tests** (including a bug we found and fixed
  where savings were understated 100×)
- App boots cleanly (health check `ok`), demo runs in ~3 s, **9/9 tests pass**

---

## 10. Honest limitations & easy next steps

- **Weather not in the model** — no temperature data in the CSV. Adding a
  temperature column would tighten forecasts further.
- **Basic rule-based agent** — the "AI" is interpretable rules over statistics,
  not an LLM. That is intentional for reliability at this level; an LLM could be
  added on top to rewrite the reasoning in natural language.
- **5-hour data resolution** — data was sampled every 5 h; the full 1M-row file
  has hourly data if you want finer forecasts (app will load a bit slower).
- **Cost rate is a single flat $/kWh** — real bills include demand charges; that
  is the natural next refinement.