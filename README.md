# Multi-Site Demand Forecasting Engine with AI Insights

An end-to-end machine learning system designed to forecast multi-location electricity demand across municipal Electric Vehicle (EV) charging networks. Built with regularized baselines, gradient-boosted regression, behavioral clustering, and automated executive summary reporting.

---

## 📌 Project Overview
* **Problem**: Public EV charging stations face unpredictable daily demand spikes, risking grid overloads and poor station availability.
* **Goal**: Build a robust, multi-site forecasting pipeline that transforms irregular transaction logs into a continuous hourly grid, clusters stations by usage behaviors, and predicts future hourly demand across sites.
* **Domain**: Municipal Electric Vehicle Infrastructure (City of Boulder Open Data).

---

## 📂 Dataset Information
* **Source**: [City of Boulder Open Data Portal - EV Charging Station Data](https://cityofboulder.maps.arcgis.com/home/item.html?id=95992b3938be4622b07f0b05eba95d4c)
* **Scope**: 2021-01-01 through 2023-12-31 (Post-lockdown EV adoption window).
* **Raw Records**: ~148,000 transaction events across city-owned charging ports.
* **Target Metric**: Electricity dispensed in kilowatt-hours (`energy_kwh`).

---

## 🏗️ Pipeline Architecture

1. **Ingestion & Data Cleaning (`src/01_clean_data.py`)**:
   - Filter to 2021–2023 window (83,725 valid sessions).
   - Filter 35 consistently active stations (minimum 150 sessions).
   - Perform hourly resampling to convert sporadic receipts into a continuous regular time-series grid (919,800 hourly observations), explicitly imputing quiet/unoccupied hours with 0.0 kWh.
2. **Exploratory Data Analysis & Visualizations (`src/02_eda.py`)**:
   - Discovered that **95.2% of station-hours have 0.0 kWh demand**, justifying **WAPE** over MAPE.
   - Identified morning commuter peak at 8:00 AM (3.0 kWh weekday avg) vs. flatter weekend leisure curves.
   - Captured 3-year adoption growth from 15 MWh/month (2021) to 35+ MWh/month (2023).
   - Generated high-res visualization figures stored in `reports/figures/`.
3. **Station Profiling & Behavioral Clustering (`src/03_cluster_stations.py`)**: *(In Progress)*
   - Extract multi-dimensional usage fingerprints (daily volume, weekday ratio, peak hour, load factor).
   - Group stations with K-Means to feed cluster context into forecasting models.
4. **Time-Series Forecasting & Multi-Tier Benchmarking (`src/04_forecast_models.py`)**: *(Upcoming)*
   - Seasonal Naive Baseline vs. Ridge Regression vs. LightGBM / Gradient Boosting.
5. **Automated AI Insights (`src/05_llm_report.py`)**: *(Upcoming)*
   - Automated plain-English operational brief for city grid operators.

---

## 🚀 Getting Started

### 1. Requirements
* Python 3.11+
* `pandas`, `numpy`, `scikit-learn`, `matplotlib`

### 2. Execution Workflow
```bash
# Step 1: Clean transactions and build continuous hourly grid
python src/01_clean_data.py

# Step 2: Run Exploratory Data Analysis and export charts
python src/02_eda.py
```
Outputs:
* Clean regularized dataset: `data/processed/hourly_station_demand.csv`
* Visualizations: `reports/figures/*.png`

