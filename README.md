# Multi-Site Electric Vehicle Demand Forecasting

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: Complete](https://img.shields.io/badge/Status-Complete-success.svg)]()
[![Dataset: City of Boulder](https://img.shields.io/badge/Dataset-City%20of%20Boulder%20EV-informational.svg)](https://open-data.bouldercolorado.gov/datasets/cityofboulder::electric-vehicle-charging-station-data/about)

An end-to-end machine learning system for regularizing, profiling, and forecasting hourly electricity load across 35 municipal Electric Vehicle (EV) charging stations in the City of Boulder, Colorado.

The platform transforms raw, uncoordinated transaction logs into a continuous Cartesian time-series grid (919,800 hourly observations), clusters charging stations into behavioral archetypes, benchmarks multi-tier predictive models under expanding walk-forward validation, and generates automated executive reports via the Google Gemini API.

---

## Executive Overview

Municipal EV charging networks present a severe operational challenge for utilities and distribution system operators: **extreme demand sparsity**. Unlike macro-level grid demand, individual public chargers sit idle for extended periods, punctuated by sharp, unscheduled charging events.

* **Sparsity:** 95.2% of all station-hours register zero energy delivery (0.0 kWh).
* **Metric Failure:** Standard percentage error metrics such as Mean Absolute Percentage Error (MAPE) divide by zero and fail on intermittent data.
* **Heterogeneity:** Stations exhibit divergent demand behavior based on urban zoning, from morning commuter lots to evening commercial districts and quiet recreational ports.

This project implements an empirical solution to multi-site load forecasting under high zero-inflation, establishing a reproducible benchmark comparing seasonal baselines, regularized linear models, and Poisson-loss gradient boosted trees.

---

## System Architecture

```
Raw Transaction Logs (148k+ events, 2021-2023)
                      |
                      v
[ 01_clean_data.py ] -> Continuous Cartesian Grid (35 stations x 26,280 hours = 919,800 rows)
                      |
                      v
[ 02_eda.py ] --------> Diurnal profiles, weekday/weekend splits, growth trends, sparsity analysis
                      |
                      v
[ 03_cluster_stations.py ] -> 4D Behavioral Fingerprints & Standardized K-Means (K=3)
                      |
                      v
[ 04_forecast_models.py ] --> 3-Fold Expanding Walk-Forward Validation (458,000+ test hours)
                              * Tier 1: Seasonal Naive Baseline (t-168)
                              * Tier 2: Ridge Regression (L2 regularization)
                              * Tier 3: Poisson Gradient Boosted Trees (Champion)
                      |
                      v
[ 05_evaluate.py ] ---------> Model residual evaluation & station error rankings
                      |
                      v
[ 06_llm_report.py ] -------> Automated LLM synthesis via Google Gemini API
                              * Executive Briefing exported to reports/EXECUTIVE_SUMMARY.md
```

---

## Benchmark Results

The forecasting models were evaluated using 3-fold expanding window Walk-Forward Validation, simulating chronological production deployments across 458,000+ out-of-sample test hours:

| Model Tier | Algorithm | WAPE (%) | MAE (kWh) | RMSE (kWh) | Description |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Tier 3** | **Gradient Boosted Trees (Champion)** | **154.10%** | **1.524** | **5.515** | Non-linear tree splits with Poisson loss, multi-hour lags (`t-1..t-4`, `t-24`, `t-168`), rolling statistics, and cluster features |
| Tier 1 | Seasonal Naive Baseline | 165.78% | 1.648 | 7.517 | Pure persistence model copying identical hour from previous week (`t-168`) |
| Tier 2 | Ridge Regression | 170.45% | 1.692 | 5.581 | Linear feature combinations with L2 regularization penalty |

### Key Findings

1. **Quantifiable Value of Machine Learning:** The champion Gradient Boosted Tree model delivers an **11.68 percentage point improvement** in WAPE over the Seasonal Naive baseline (a 7.0% relative improvement in total load allocation).
2. **26.6% Reduction in Catastrophic Errors:** Gradient Boosted Trees reduced Root Mean Squared Error (RMSE) from 7.517 kWh down to 5.515 kWh. Because RMSE squares errors before averaging, this 26.6% reduction reflects a dramatic drop in severe peak-demand prediction blunders, mitigating local transformer overload risk.
3. **Failure of Standard Linear Regression:** Ridge Regression exhibited the highest WAPE (170.45%). Under extreme zero-inflation, linear models apply an unconstrained continuous shift that predicts fractional "background buzz" across thousands of genuinely idle hours, accumulating substantial total error.

---

## Core Engineering Decisions

### 1. Cartesian Grid Regularization & Zero Imputation
Raw EV transaction logs record irregular timestamps (arrival and disconnect times). To enable valid time-series forecasting with stationary lag steps (`t-1`, `t-24`, `t-168`), we construct a complete Cartesian product:
$$\text{Total Observations} = 35 \text{ stations} \times 26,280 \text{ hours} = 919,800 \text{ rows}$$
All station-hours lacking active charging sessions are explicitly imputed with `0.0 kWh`.

### 2. Metric Selection: WAPE over MAPE
Given 95.2% zero values, classical MAPE:
$$\text{MAPE} = \frac{1}{N}\sum \left|\frac{y - \hat{y}}{y}\right|$$
is undefined due to division by zero. Adding arbitrary constants ($\epsilon$) artificially skews scores based on chosen epsilon scale. We utilize Weighted Absolute Percentage Error (WAPE):
$$\text{WAPE} = \frac{\sum_{i=1}^{N} |y_i - \hat{y}_i|}{\sum_{i=1}^{N} y_i}$$
WAPE weights errors proportionally by actual delivered energy, prioritizing high-load peak hours over quiet overnight intervals.

### 3. Poisson Deviance Loss Function
To combat non-negativity constraints and severe intermittency, Tier 3 Gradient Boosted Trees utilize a Poisson deviance loss:
$$\text{Loss}(y, \hat{y}) = 2 \left( y \log \frac{y}{\hat{y}} - y + \hat{y} \right)$$
When actual demand is zero ($y=0$), the loss simplifies to $2\hat{y}$. This introduces an asymmetric penalty that directly suppresses false-positive predictions during quiet hours, eliminating phantom baseline load.

### 4. Station Behavioral Profiling & K-Means Clustering
Stations were mapped into a 4-dimensional normalized feature space (`daily_volume_kwh`, `weekday_share`, `peak_hour`, `load_factor`). Using standardized K-Means ($K=3$), three distinct operating archetypes were identified:
* **High-Traffic Commuter Hubs (7 stations):** ~59.1 kWh/day average, sharp 8:00 AM arrival peak, accounts for over 54% of total municipal load (167.8 MWh in 2023). Most predictable archetype (**142.89% WAPE**).
* **Afternoon / Evening Hubs (9 stations):** ~15.1 kWh/day average, distinct 4:00 PM peak corresponding to commercial and recreational departures (**181.99% WAPE**).
* **Neighborhood / Community Ports (19 stations):** ~10.9 kWh/day average, dispersed low-utilization chargers with intermittent patterns (**165.69% WAPE**).

---

## Automated Executive Reporting

The system includes an automated intelligence layer (`src/06_llm_report.py`) that consumes pipeline artifacts (validation metrics, archetype statistics, station error rankings) and synthesizes a professional executive briefing via the Google Gemini API (model `gemini-3.6-flash`).

The briefing translates quantitative metrics into actionable grid operations strategies, including targeted Battery Energy Storage System (BESS) sizing for top commuter hubs and Time-of-Use (TOU) tariff shaping for afternoon stations. The generated report is exported to [`reports/EXECUTIVE_SUMMARY.md`](reports/EXECUTIVE_SUMMARY.md).

---

## Project Structure

```
Multi-Site-Demand-Forecasting/
├── data/
│   ├── raw/                           # Raw Boulder transaction CSVs (git-ignored)
│   └── processed/
│       └── station_clusters.csv       # Station cluster metadata and archetypes
├── reports/
│   ├── figures/                       # Publication-ready visualizations
│   │   ├── 01_diurnal_hourly_profile.png
│   │   ├── 02_weekday_vs_weekend.png
│   │   ├── 03_monthly_trend_growth.png
│   │   ├── 04_station_demand_disparity.png
│   │   ├── 05_station_clusters.png
│   │   ├── 06_forecast_vs_actual.png
│   │   ├── 07_residual_analysis.png
│   │   └── 08_station_error_rankings.png
│   ├── benchmark_results.csv          # Walk-forward model scorecard
│   ├── station_performance.csv        # 35-station accuracy rankings
│   └── EXECUTIVE_SUMMARY.md           # LLM-generated operational brief
├── src/
│   ├── 01_clean_data.py               # Ingestion and Cartesian hourly resampling
│   ├── 02_eda.py                      # Exploratory data analysis suite
│   ├── 03_cluster_stations.py         # K-Means behavioral clustering
│   ├── 04_forecast_models.py          # Multi-tier walk-forward benchmark
│   ├── 05_evaluate.py                 # Out-of-sample diagnostics & residual analysis
│   └── 06_llm_report.py               # Automated AI executive briefing generator
├── .env.example                       # Template for API configuration
├── .gitignore                         # Data and secret exclusion rules
├── LICENSE                            # MIT License
├── README.md                          # Production project documentation
└── requirements.txt                   # Dependency specifications
```

---

## Getting Started

### Prerequisites

* Python 3.9 or higher
* Recommended environment: Virtual environment (`venv` or `conda`)

### Installation

```bash
# Clone the repository
git clone https://github.com/arunabh2005/Multi-Site-Demand-Forecasting.git
cd Multi-Site-Demand-Forecasting

# Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` to enable live LLM report generation (optional):

```bash
# Set your Gemini API key (free tier supported)
GEMINI_API_KEY=your_gemini_api_key_here
```

*Note: If no API key is provided, `src/06_llm_report.py` will run in local deterministic mode and generate a complete report using embedded analytical templates.*

### Running the Pipeline

Execute each pipeline stage sequentially:

```bash
# Step 1: Regularize transaction logs into a continuous hourly grid
python src/01_clean_data.py

# Step 2: Generate exploratory visualizations and sparsity analysis
python src/02_eda.py

# Step 3: Cluster stations into behavioral archetypes
python src/03_cluster_stations.py

# Step 4: Run expanding walk-forward multi-tier model benchmark
python src/04_forecast_models.py

# Step 5: Run model diagnostics and station rankings
python src/05_evaluate.py

# Step 6: Generate executive summary brief
python src/06_llm_report.py
```

---

## References & Acknowledgments

* **City of Boulder Open Data Portal**: *Electric Vehicle Charging Station Data*, City of Boulder, Colorado ([Dataset Link](https://open-data.bouldercolorado.gov/datasets/cityofboulder::electric-vehicle-charging-station-data/about)).
* **Intermittent Demand & Forecast Metric Literature**:
  * Syntetos, A. A., & Boylan, J. E. (2005). *The accuracy of intermittent demand estimates*. International Journal of Forecasting, 21(2), 303-314.
  * Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679-688.
* **Poisson Regression & Gradient Boosting**:
  * Nelder, J. A., & Wedderburn, R. W. (1972). *Generalized Linear Models*. Journal of the Royal Statistical Society: Series A (General), 135(3), 370-384.
  * Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine*. Annals of Statistics, 29(5), 1189-1232.

---

## Author

**Arunabh Das** ([LinkedIn](https://www.linkedin.com/in/arunabh-das-ba9a1725b/))
