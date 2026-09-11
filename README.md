# Multi-Site EV Demand Forecasting

A time-series forecasting project to predict hourly electricity demand across municipal Electric Vehicle (EV) charging stations in Boulder, Colorado. 

The goal is to move from raw, sporadic charging transaction logs to reliable multi-station load forecasts, evaluating whether machine learning adds measurable value over standard statistical baselines.

---

## The Problem

Public EV chargers do not behave like a smooth, continuous power grid. Individual stations experience heavy intermittency: chargers sit completely idle for hours at a time, followed by sharp spikes when commuters plug in. 

Because of this, standard textbook approaches often break down:
* Over **95% of individual station-hours have 0.0 kWh demand**.
* Traditional percentage metrics like MAPE divide by zero and explode on quiet hours.
* Stations have distinct behavioral profiles (downtown commuter lots behave differently from public parks and rec centers).

---

## Data Source

The project uses open transaction records provided by the City of Boulder Open Data Portal:
* **Portal Page:** [City of Boulder EV Charging Station Data](https://open-data.bouldercolorado.gov/datasets/cityofboulder::electric-vehicle-charging-station-data/about) (ArcGIS Hub Item: `95992b3938be4622b07f0b05eba95d4c`)
* **Direct CSV Download:** [Download raw Boulder EV CSV](https://open-data.bouldercolorado.gov/api/download/v1/items/95992b3938be4622b07f0b05eba95d4c/csv?layers=0)
* **Local Location:** Saved as `data/raw/boulder_ev_charging.csv`
* **Timeframe:** January 1, 2021 – December 31, 2023 (3 full calendar years).
* **Stations tracked:** 35 active municipal charging stations (filtered for consistent multi-year history).
* **Target variable:** Total energy delivered in kilowatt-hours (`energy_kwh`) per station per hour.


---

## Current Progress & Pipeline

### 1. Ingestion and Time-Series Regularization (`src/01_clean_data.py`)
* Ingests ~148,000 raw charging events and filters to valid positive sessions between 2021 and 2023.
* Builds a continuous Cartesian grid across all 35 stations for all 26,280 hours in the 3-year period (919,800 total station-hour rows).
* Explicitly fills quiet and overnight hours with `0.0 kWh` to ensure predictable lag steps (`t-24`, `t-168`) for time-series modeling.

### 2. Exploratory Data Analysis (`src/02_eda.py`)
Generates exploratory visualizations saved to `reports/figures/`:
* **Diurnal load curve:** Identifies a sharp morning commute spike at 8:00 AM (~3.0 kWh weekday avg) and an afternoon plateau between 10:00 AM and 2:00 PM.
* **Weekday vs. Weekend split:** Highlights heavy commuter dependency during the workweek compared to delayed, flatter weekend charging patterns.
* **Macro trend:** Tracks total network consumption growth from ~15 MWh/month in early 2021 to over 35 MWh/month by late 2023.
* **Metric selection:** Confirms that Weighted Absolute Percentage Error (WAPE) must be used over MAPE to handle sparse zero-demand hours.

### 3. Station Behavioral Profiling & Clustering (`src/03_cluster_stations.py`)
Extracts a 4-dimensional behavioral fingerprint for all 35 stations and applies standardized K-Means clustering ($K=3$):
* **High-Traffic Commuter Hubs (7 stations):** ~59.1 kWh/day average, sharp 8:00 AM arrival peak.
* **Afternoon / Evening Hubs (9 stations):** ~15.1 kWh/day average, prominent 4:00 PM peak (post-work dining/shopping).
* **Neighborhood / Community Ports (19 stations):** ~10.9 kWh/day average, quiet local charging spots.
* Outputs cluster mapping to `data/processed/station_clusters.csv` and cluster profiles to `reports/figures/05_station_clusters.png`.

### 4. Next Steps
* **Multi-Tier Forecasting (`src/04_forecast_models.py`):** Benchmark Seasonal Naive Baseline vs. Ridge Regression vs. LightGBM on a held-out temporal test set (using station cluster features).
* **Automated Operations Summary (`src/05_llm_report.py`):** Generate plain-English executive takeaways using an LLM.


---

## Project Structure

```
├── data/
│   ├── raw/                 # Raw Boulder transaction CSVs
│   └── processed/           # Continuous hourly grid (generated)
├── reports/
│   └── figures/             # Exported EDA and evaluation plots
├── src/
│   ├── 01_clean_data.py     # Ingestion and hourly resampling
│   ├── 02_eda.py            # Exploratory data analysis suite
│   ├── 03_cluster_stations.py
│   ├── 04_forecast_models.py
│   └── 05_llm_report.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## How to Run

### Setup
```bash
# Clone the repository
git clone https://github.com/arunabh2005/Multi-Site-Demand-Forecasting.git
cd Multi-Site-Demand-Forecasting

# Install dependencies
pip install -r requirements.txt
```

### Execution
```bash
# Step 1: Clean transactions and build hourly continuous grid
python src/01_clean_data.py

# Step 2: Run EDA and generate figures
python src/02_eda.py

# Step 3: Run station clustering and extract behavioral archetypes
python src/03_cluster_stations.py
```

