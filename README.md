# Multi-Site EV Demand Forecasting

A time-series forecasting project to predict hourly electricity demand across municipal Electric Vehicle (EV) charging stations in Boulder, Colorado. 

The goal is to move from raw, sporadic charging transaction logs to reliable multi-station load forecasts, evaluating whether machine learning adds measurable value over standard statistical baselines.

> **Engineering & Concept Reference:** For detailed interview explanations on time-series regularization, metric selection (WAPE vs. MAPE), walk-forward validation, and lag engineering, see [**`ML_CONCEPTS_AND_INTERVIEWS.md`**](ML_CONCEPTS_AND_INTERVIEWS.md).


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

### 4. Multi-Tier Forecasting & Walk-Forward Validation (`src/04_forecast_models.py`)
Evaluates 3 model tiers using 3-fold expanding window Walk-Forward Validation (simulating periodic production retrainings across 2021–2023 with 458,000+ test hours):

| Model Tier | WAPE (%) | MAE (kWh) | RMSE (kWh) | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Tier 3: Gradient Boosted Trees (Champion)** | **154.10%** | **1.524** | **5.515** | Non-linear tree splits with Poisson loss, multi-hour lags (`t-1..t-4`, `t-24`, `t-168`), and cluster context |
| **Tier 1: Seasonal Naive (Baseline 1)** | 165.78% | 1.648 | 7.517 | Copy-paste rule (same hour last week $t-168$) |
| **Tier 2: Ridge Regression (Baseline 2)** | 170.45% | 1.692 | 5.581 | Linear model with L2 regularization |

* **Main Result:** Gradient Boosted Trees outperforms Seasonal Naive by **11.68 percentage points** (7.0% relative gain) and slashes catastrophic spike blunders (RMSE) by **26.6%** over the naive baseline.
* Outputs scorecard to `reports/benchmark_results.csv` and 7-day forecast comparison to `reports/figures/06_forecast_vs_actual.png`.

### 5. Next Steps
* **Out-of-Sample Diagnostics & Residuals (`src/05_evaluate.py`):** Deep-dive into station-by-station error distributions and identify which locations are easiest vs hardest to predict.
* **Automated AI Insights (`src/06_llm_report.py`):** Generate natural language executive operational briefs using an LLM.



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
├── ML_CONCEPTS_AND_INTERVIEWS.md
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

# Step 4: Run Walk-Forward multi-tier forecasting benchmark
python src/04_forecast_models.py
```


