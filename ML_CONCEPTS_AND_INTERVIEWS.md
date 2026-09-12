# Machine Learning & Time-Series Interview Guide

A practical reference guide of core concepts, real-world data traps, and interview-ready talking points derived from this project. 

Use this guide to review the technical "why" behind each architectural decision and to explain the project with confidence in interviews.

---

## 1. Time-Series Regularization & Imputation
* **Where in code:** `src/01_clean_data.py`
* **The Problem:** Raw EV transactions are sporadic event logs. If nobody charges between 1:00 AM and 5:00 AM, there are zero rows in the raw CSV.
* **The Solution:** We constructed a full Cartesian product grid (`35 stations` × `26,280 hours`), ensuring exactly 24 hourly rows per day per station, and filled quiet hours with `0.0 kWh`.
* **Why it matters:** 
  1. Missing rows and zero demand are fundamentally different in physics and ML. A charger drawing 0.0 kW is not "missing data"—it is real, operational truth.
  2. A fixed 1-hour interval guarantees that looking back 24 hours ($t-24$) or 1 week ($t-168$) is strictly fixed index math.
* **Interview Soundbite:**  
  > *"Real-world event logs only record actions, not inaction. By regularizing the sporadic transactions into a continuous 24-hour hourly grid with explicit zero-imputation, we turned irregular transaction receipts into a dependable time-series with deterministic lag alignment."*

---

## 2. Metric Selection: WAPE vs. MAPE (The Zero-Division Trap)
* **Where in code:** `src/02_eda.py`, `PROGRESS.md`
* **The Problem:** In our exploratory analysis, we discovered that **95.2% of all station-hours have 0.0 kWh demand**.
* **Why MAPE fails:** Mean Absolute Percentage Error calculates `|actual - pred| / actual` for each hour. When `actual == 0`, it divides by zero and explodes to infinity. Even on tiny values (e.g., actual = 1, pred = 3), MAPE reports a misleading 200% error.
* **The Solution:** Weighted Absolute Percentage Error (WAPE):
  $$\text{WAPE} = \frac{\sum |\text{actual} - \text{predicted}|}{\sum \text{actual}}$$
* **Interview Soundbite:**  
  > *"Textbook MAPE fails on intermittent, zero-inflated demand because 95% of individual station hours are zero. During EDA, we caught this early and chose WAPE instead. WAPE aggregates total errors over total network demand, meaning it never divides by zero and naturally weights high-traffic peak hours over trivial 3:00 AM noise."*

---

## 3. Multi-Dimensional Behavioral Clustering (K-Means)
* **Where in code:** `src/03_cluster_stations.py`
* **The Problem:** Training 35 individual models is an operational maintenance headache. Training one generic model ignores the fact that downtown commuter garages behave completely differently from public parks.
* **The Solution:** We compressed 3 years of hourly load for each station into a 4D behavioral fingerprint:
  1. `daily_volume_kwh`: Total daily load scale.
  2. `weekday_share`: Proportion of energy on Mon-Fri (avoids ratio division-by-zero).
  3. `peak_hour`: Hour of maximum surge.
  4. `load_factor`: Mean load / peak load (demand consistency).
* **Why `StandardScaler` was required:** Volume is measured in tens of kWh, while weekday share is between 0.0 and 1.0. Without standardizing to mean=0, std=1, volume would overpower the distance calculation.
* **The Result:** Grouped 35 stations into 3 archetypes: *High-Traffic Commuter Hubs* (7), *Afternoon/Evening Hubs* (9), and *Neighborhood/Community Ports* (19).
* **Interview Soundbite:**  
  > *"Rather than maintaining 35 separate models or one blind global model, we extracted standardized behavioral fingerprints and applied K-Means clustering. Passing the cluster archetype as a feature allows a single gradient-boosted model to share cross-station learning while still tailoring predictions to each station's operational archetype."*

---

## 4. Calendar Signals vs. Autoregressive Lags
* **Where in code:** `src/04_forecast_models.py`
* **Calendar Features (The Routine):** `hour`, `day_of_week`, `is_weekend`, `month`.  
  * Captures the regular rhythm of human life (e.g., commuters arrive at 8:00 AM on weekdays).
* **Lag Features (The Momentum / Rearview Mirror):**
  * `t - 1`: Immediate momentum (a car that plugged in 1 hour ago is often still plugged in).
  * `t - 24`: Yesterday at the exact same hour.
  * `t - 168`: Same day and hour last week ($7 \times 24 = 168$).
* **Rolling Window Statistics:** 24-hour and 7-day rolling means to smooth out short-term fluctuations and capture local trend velocity.
* **Interview Soundbite:**  
  > *"Calendar features inform the model of expected human schedules, while autoregressive lags provide immediate operational context. If an unexpected weather event hits Boulder, calendar features would predict normal traffic, but the t-1 and t-24 lags immediately alert the model that actual demand dropped to zero."*

---

## 5. Avoiding Data Leakage: Why Random K-Fold is Forbidden
* **Where in code:** `src/04_forecast_models.py`
* **The Trap:** Randomly shuffling rows into cross-validation folds causes **Lookahead Leakage (time-travel)**. The model trains on rows from November 2023 to predict April 2021.
* **The Consequence:** The model memorizes future patterns, resulting in unrealistically high test scores in development that catastrophically fail when deployed to production.
* **The Rule:** Time must only flow forward. Training data must strictly precede testing data.
* **Interview Soundbite:**  
  > *"Random K-Fold cross-validation in time-series causes lookahead leakage by training on tomorrow to predict yesterday. We enforce strict temporal boundaries where training data chronologically precedes evaluation data, ensuring test metrics reflect real-world operational performance."*

---

## 6. Walk-Forward Validation (Backtesting / TimeSeriesSplit)
* **Where in code:** `src/04_forecast_models.py`
* **What it is:** Multi-fold validation adapted for sequential time series using an **expanding window**:
  * **Fold 1:** Train on [2021] ➔ Test on [Early 2022]
  * **Fold 2:** Train on [2021 – Early 2022] ➔ Test on [Late 2022]
  * **Fold 3:** Train on [2021 – 2022] ➔ Test on [2023]
* **Why it is superior to a single train/test split:**
  1. Evaluates the model across multiple seasonal transitions (winter snowstorms, summer heat, changing EV adoption rates).
  2. Proves that model accuracy is consistent and robust over time, not just lucky on one specific test month.
* **Interview Soundbite:**  
  > *"A single holdout split only tells you how your model performed during one specific season. By implementing Walk-Forward Validation using an expanding window, we simulated quarterly retrainings across the multi-year history, ensuring the model's accuracy was stable across varying adoption cycles."*

---

## 7. Multi-Tier Benchmarking (Why We Need Baselines)
* **Where in code:** `src/04_forecast_models.py`
* **The 3-Tier Hierarchy:**
  1. **Tier 1 — Simple Baseline (Seasonal Naive):** Predicts whatever happened at the same station at the same hour last week ($t-168$). Takes 0 seconds to train.
  2. **Tier 2 — Moderate Baseline (Ridge Regression):** Linear model with L2 regularization using calendar indicators and lags.
  3. **Tier 3 — Main ML Model (HistGradientBoosting / LightGBM):** Non-linear gradient-boosted decision trees using lag features, rolling statistics, calendar indicators, and station cluster labels.
* **Why Baselines are Mandatory:** An ML engineer who only presents a complex model with "80% accuracy" doesn't know if their model is actually good. If a zero-cost rule like Seasonal Naive achieves 82%, the complex model is worse than useless.
* **Interview Soundbite:**  
  > *"In production ML, complexity must earn its way. We established a strict three-tier benchmark: Seasonal Naive as our lower bound, Ridge Regression as our linear baseline, and Gradient Boosted Trees as our candidate model. This transparently proves exactly how much performance gain was bought by adding feature engineering and tree-based non-linearities."*
