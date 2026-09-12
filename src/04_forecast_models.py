"""
04_forecast_models.py
---------------------
Phase 4: Time-Series Feature Engineering, Walk-Forward Validation, and Multi-Tier Benchmarking.

1. Loads hourly demand (data/processed/hourly_station_demand.csv) and merges station clusters.
2. Constructs feature space:
   - Calendar features: hour, dayofweek, is_weekend, month.
   - Autoregressive Lags: t-1, t-2, t-24 (yesterday same hour), t-168 (last week same hour).
   - Rolling window momentum: 24h rolling mean, 168h rolling mean (shifted to prevent leakage).
   - Categorical metadata: cluster_id.
3. Implements Walk-Forward Validation (Expanding Window / TimeSeriesSplit):
   - Fold 1: Train 2021 ➔ Test H1 2022
   - Fold 2: Train 2021-2022 ➔ Test H1 2023
   - Fold 3: Train 2021-H1 2023 ➔ Test H2 2023
4. Evaluates 3 Tiers of Models:
   - Tier 1: Simple Baseline (Seasonal Naive: last week same hour)
   - Tier 2: Moderate Baseline (Ridge Regression with L2 regularization)
   - Tier 3: Main Model (HistGradientBoostingRegressor with non-linear tree splits)
5. Computes WAPE (%), MAE (kWh), and RMSE (kWh).
6. Exports benchmark scorecard to reports/benchmark_results.csv.
7. Plots 7-day actual vs. predicted load curve to reports/figures/06_forecast_vs_actual.png.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEMAND_FILE = "data/processed/hourly_station_demand.csv"
CLUSTERS_FILE = "data/processed/station_clusters.csv"
OUTPUT_REPORT_FILE = "reports/benchmark_results.csv"
OUTPUT_FIG_PATH = "reports/figures/06_forecast_vs_actual.png"

def calculate_wape(actual, predicted):
    """Weighted Absolute Percentage Error: sum(|actual - pred|) / sum(actual) * 100"""
    sum_actual = np.sum(actual)
    if sum_actual == 0:
        return 0.0
    return (np.sum(np.abs(actual - predicted)) / sum_actual) * 100.0

def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def build_features(df_raw, df_clusters):
    print("Building time-series features (calendar, lags, rolling momentum)...")
    df = df_raw.copy()
    df['hourly_timestamp'] = pd.to_datetime(df['hourly_timestamp'])
    df = df.sort_values(['station_name', 'hourly_timestamp']).reset_index(drop=True)

    # 1. Calendar Features
    df['hour'] = df['hourly_timestamp'].dt.hour
    df['dayofweek'] = df['hourly_timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['month'] = df['hourly_timestamp'].dt.month

    # 2. Attach Station Clusters
    if df_clusters is not None:
        df = df.merge(df_clusters[['station_name', 'cluster_id']], on='station_name', how='left')
        df['cluster_id'] = df['cluster_id'].fillna(0).astype(int)
    else:
        df['cluster_id'] = 0

    # 3. Autoregressive Lag Features (grouped per station)
    # Strictly shifted backwards so no lookahead
    grouped = df.groupby('station_name')['energy_kwh']
    df['lag_1'] = grouped.shift(1)
    df['lag_2'] = grouped.shift(2)
    df['lag_3'] = grouped.shift(3)  # Captures ongoing multi-hour sessions
    df['lag_4'] = grouped.shift(4)
    df['lag_24'] = grouped.shift(24)
    df['lag_168'] = grouped.shift(168)  # 1 week ago same hour (Seasonal Naive)

    # 4. Rolling Statistics (shifted by 1 so current hour actual is excluded!)
    df['rolling_mean_24'] = grouped.transform(lambda x: x.shift(1).rolling(24, min_periods=6).mean())
    df['rolling_max_24'] = grouped.transform(lambda x: x.shift(1).rolling(24, min_periods=6).max())
    df['rolling_mean_168'] = grouped.transform(lambda x: x.shift(1).rolling(168, min_periods=24).mean())

    # Drop the warmup window (first 168 hours per station will have NaNs from lag_168)
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Features created. Dropped warmup window ({initial_len - len(df):,} rows). Valid rows: {len(df):,}")

    return df


def run_benchmarking():
    print("=" * 65)
    print("STEP 4: MULTI-TIER FORECASTING & WALK-FORWARD BENCHMARKING")
    print("=" * 65)

    # 1. Load data
    print(f"Loading continuous hourly demand: {DEMAND_FILE}")
    df_raw = pd.read_csv(DEMAND_FILE)
    df_clusters = pd.read_csv(CLUSTERS_FILE) if os.path.exists(CLUSTERS_FILE) else None

    # 2. Engineer features
    df = build_features(df_raw, df_clusters)

    feature_cols = [
        'hour', 'dayofweek', 'is_weekend', 'month', 'cluster_id',
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_24', 'lag_168',
        'rolling_mean_24', 'rolling_max_24', 'rolling_mean_168'
    ]
    target_col = 'energy_kwh'

    # 3. Define Walk-Forward Folds (Expanding Windows)
    # Fold 1: Train 2021 -> Test H1 2022 (Jan-Jun 2022)
    # Fold 2: Train 2021-2022 -> Test H1 2023 (Jan-Jun 2023)
    # Fold 3: Train 2021-H1 2023 -> Test H2 2023 (Jul-Dec 2023)
    folds = [
        {
            "name": "Fold 1 (Test H1 2022)",
            "train_end": pd.Timestamp("2021-12-31 23:00:00"),
            "test_start": pd.Timestamp("2022-01-01 00:00:00"),
            "test_end": pd.Timestamp("2022-06-30 23:00:00")
        },
        {
            "name": "Fold 2 (Test H1 2023)",
            "train_end": pd.Timestamp("2022-12-31 23:00:00"),
            "test_start": pd.Timestamp("2023-01-01 00:00:00"),
            "test_end": pd.Timestamp("2023-06-30 23:00:00")
        },
        {
            "name": "Fold 3 (Test H2 2023)",
            "train_end": pd.Timestamp("2023-06-30 23:00:00"),
            "test_start": pd.Timestamp("2023-07-01 00:00:00"),
            "test_end": pd.Timestamp("2023-12-31 23:00:00")
        }
    ]

    results_records = []
    final_test_predictions = None

    print("\n" + "-" * 65)
    print("RUNNING WALK-FORWARD VALIDATION (EXPANDING WINDOWS)")
    print("-" * 65)

    for f_idx, fold in enumerate(folds):
        print(f"\n--- {fold['name']} ---")
        train_mask = df['hourly_timestamp'] <= fold['train_end']
        test_mask = (df['hourly_timestamp'] >= fold['test_start']) & (df['hourly_timestamp'] <= fold['test_end'])

        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target_col]
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, target_col]

        print(f"  Train: {X_train.shape[0]:,} rows (up to {fold['train_end'].date()})")
        print(f"  Test:  {X_test.shape[0]:,} rows ({fold['test_start'].date()} to {fold['test_end'].date()})")

        # --- MODEL 1: Tier 1 - Seasonal Naive Baseline ---
        # Predicts lag_168 (same hour last week)
        y_pred_naive = np.maximum(X_test['lag_168'].values, 0.0)
        wape_naive = calculate_wape(y_test.values, y_pred_naive)
        mae_naive = mean_absolute_error(y_test.values, y_pred_naive)
        rmse_naive = calculate_rmse(y_test.values, y_pred_naive)
        print(f"  [Tier 1] Seasonal Naive        | WAPE: {wape_naive:6.2f}% | MAE: {mae_naive:.3f} kWh | RMSE: {rmse_naive:.3f}")

        results_records.append({
            'fold': fold['name'], 'model': 'Tier 1: Seasonal Naive',
            'wape_pct': round(wape_naive, 2), 'mae_kwh': round(mae_naive, 3), 'rmse_kwh': round(rmse_naive, 3)
        })

        # --- MODEL 2: Tier 2 - Ridge Regression ---
        t0 = time.time()
        ridge = Ridge(alpha=10.0, random_state=42)
        ridge.fit(X_train, y_train)
        y_pred_ridge = np.maximum(ridge.predict(X_test), 0.0) # EV demand cannot be negative
        fit_time_ridge = time.time() - t0

        wape_ridge = calculate_wape(y_test.values, y_pred_ridge)
        mae_ridge = mean_absolute_error(y_test.values, y_pred_ridge)
        rmse_ridge = calculate_rmse(y_test.values, y_pred_ridge)
        print(f"  [Tier 2] Ridge Regression      | WAPE: {wape_ridge:6.2f}% | MAE: {mae_ridge:.3f} kWh | RMSE: {rmse_ridge:.3f} (fit: {fit_time_ridge:.1f}s)")

        results_records.append({
            'fold': fold['name'], 'model': 'Tier 2: Ridge Regression',
            'wape_pct': round(wape_ridge, 2), 'mae_kwh': round(mae_ridge, 3), 'rmse_kwh': round(rmse_ridge, 3)
        })

        # --- MODEL 3: Tier 3 - Gradient Boosted Trees (Poisson Loss) ---
        t0 = time.time()
        hgb = HistGradientBoostingRegressor(
            loss='poisson',
            max_iter=150,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            categorical_features=['cluster_id', 'is_weekend'],
            random_state=42
        )
        hgb.fit(X_train, y_train)
        y_pred_hgb = np.maximum(hgb.predict(X_test), 0.0)

        fit_time_hgb = time.time() - t0

        wape_hgb = calculate_wape(y_test.values, y_pred_hgb)
        mae_hgb = mean_absolute_error(y_test.values, y_pred_hgb)
        rmse_hgb = calculate_rmse(y_test.values, y_pred_hgb)
        print(f"  [Tier 3] Gradient Boosted Trees| WAPE: {wape_hgb:6.2f}% | MAE: {mae_hgb:.3f} kWh | RMSE: {rmse_hgb:.3f} (fit: {fit_time_hgb:.1f}s)")

        results_records.append({
            'fold': fold['name'], 'model': 'Tier 3: Gradient Boosted Trees',
            'wape_pct': round(wape_hgb, 2), 'mae_kwh': round(mae_hgb, 3), 'rmse_kwh': round(rmse_hgb, 3)
        })

        # Keep fold 3 test predictions for visualization
        if f_idx == len(folds) - 1:
            final_test_df = df.loc[test_mask, ['station_name', 'hourly_timestamp']].copy()
            final_test_df['actual'] = y_test.values
            final_test_df['naive_pred'] = y_pred_naive
            final_test_df['ridge_pred'] = y_pred_ridge
            final_test_df['gbt_pred'] = y_pred_hgb
            final_test_predictions = final_test_df

    # 4. Summary Scorecard Across All Folds
    results_df = pd.DataFrame(results_records)
    summary_scorecard = results_df.groupby('model')[['wape_pct', 'mae_kwh', 'rmse_kwh']].mean().reset_index()
    summary_scorecard = summary_scorecard.sort_values('wape_pct', ascending=True)

    print("\n" + "=" * 65)
    print("WALK-FORWARD BENCHMARK SCORECARD (AVERAGE OVER ALL FOLDS):")
    print("=" * 65)
    for _, row in summary_scorecard.iterrows():
        print(f"  {row['model']:<30} | WAPE: {row['wape_pct']:5.2f}% | MAE: {row['mae_kwh']:.3f} kWh | RMSE: {row['rmse_kwh']:.3f}")

    # Calculate Improvement
    naive_wape = summary_scorecard.loc[summary_scorecard['model'].str.contains('Seasonal Naive'), 'wape_pct'].values[0]
    gbt_wape = summary_scorecard.loc[summary_scorecard['model'].str.contains('Gradient Boosted'), 'wape_pct'].values[0]
    improvement_pts = naive_wape - gbt_wape
    improvement_rel = (improvement_pts / naive_wape) * 100.0

    print("-" * 65)
    print(f" MAIN MODEL PERFORMANCE GAIN:")
    print(f"  Gradient Boosted Trees beats Seasonal Naive by {improvement_pts:.2f} percentage points!")
    print(f"  Relative error reduction: {improvement_rel:.1f}%")
    print("=" * 65)

    # Save benchmark table
    os.makedirs("reports", exist_ok=True)
    summary_scorecard.to_csv(OUTPUT_REPORT_FILE, index=False)
    print(f"\n[SUCCESS] Benchmark scorecard saved to: {OUTPUT_REPORT_FILE}")

    # 5. Export Actual vs. Forecast Visualization
    print(f"Generating Forecast vs. Actual plot: {OUTPUT_FIG_PATH}...")
    # Pick the top commuter station (N BOULDER REC 1 or BASELINE ST1) over a 7-day window (Oct 2-8, 2023)
    sample_station = "BOULDER / BASELINE ST1"
    station_sub = final_test_predictions[final_test_predictions['station_name'] == sample_station].copy()
    
    # Filter to 1 full week
    sample_week = station_sub[
        (station_sub['hourly_timestamp'] >= "2023-10-02 00:00:00") & 
        (station_sub['hourly_timestamp'] <= "2023-10-08 23:00:00")
    ].sort_values('hourly_timestamp')

    if len(sample_week) > 0:
        fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
        ax.plot(sample_week['hourly_timestamp'], sample_week['actual'], color='#111111', linewidth=2.2, label='Actual Demand (Ground Truth)', zorder=4)
        ax.plot(sample_week['hourly_timestamp'], sample_week['gbt_pred'], color='#2ca02c', linewidth=2.0, linestyle='--', label='Gradient Boosted Trees (Main Model)', zorder=3)
        ax.plot(sample_week['hourly_timestamp'], sample_week['naive_pred'], color='#d62728', linewidth=1.2, linestyle=':', alpha=0.75, label='Seasonal Naive Baseline', zorder=2)

        ax.set_title(f'Held-Out Test Forecast vs. Actual: {sample_station} (Oct 2 - Oct 8, 2023)', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel('Timestamp (1-Hour Resolution)', fontsize=10)
        ax.set_ylabel('Electricity Demand (kWh)', fontsize=10)
        ax.legend(frameon=True, facecolor='white', loc='upper right')
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUTPUT_FIG_PATH)
        plt.close(fig)
        print(f" -> Saved: {OUTPUT_FIG_PATH}")
    else:
        print("Warning: Sample week slice was empty; check date ranges.")

    print("=" * 65)

if __name__ == "__main__":
    run_benchmarking()
