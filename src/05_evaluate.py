"""
05_evaluate.py
--------------
Phase 5: Out-of-Sample Diagnostics, Residual Analysis, and Station Error Rankings.

1. Trains the winning Gradient Boosted Trees model (with Poisson loss and cluster features)
   on 2021-2022 data (~607,000 rows).
2. Generates out-of-sample forecasts on the full held-out test year of 2023 (~306,000 rows).
3. Evaluates station-by-station performance:
   - Identifies the top 5 most predictable stations vs. top 5 most erratic stations.
   - Exports complete station scorecard to reports/station_performance.csv.
4. Performs Residual Analysis (Actual - Predicted):
   - Computes bias / mean residual error across 24 hours.
   - Verifies error distribution symmetry around zero.
   - Generates reports/figures/07_residual_analysis.png.
5. Generates Station Error Rankings chart:
   - Horizontal bar chart in reports/figures/08_station_error_rankings.png.
6. Evaluates Cluster-Level performance (Commuter vs. Afternoon vs. Neighborhood).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEMAND_FILE = "data/processed/hourly_station_demand.csv"
CLUSTERS_FILE = "data/processed/station_clusters.csv"
OUTPUT_STATION_CSV = "reports/station_performance.csv"
FIG_RESIDUALS_PATH = "reports/figures/07_residual_analysis.png"
FIG_RANKINGS_PATH = "reports/figures/08_station_error_rankings.png"

def calculate_wape(actual, predicted):
    sum_actual = np.sum(actual)
    if sum_actual == 0:
        return 0.0
    return (np.sum(np.abs(actual - predicted)) / sum_actual) * 100.0

def run_evaluation():
    print("=" * 65)
    print("STEP 5: OUT-OF-SAMPLE DIAGNOSTICS & RESIDUAL ANALYSIS")
    print("=" * 65)

    # 1. Load data & clusters
    print(f"Loading data from {DEMAND_FILE} and clusters from {CLUSTERS_FILE}...")
    df = pd.read_csv(DEMAND_FILE)
    df_clusters = pd.read_csv(CLUSTERS_FILE)

    df['hourly_timestamp'] = pd.to_datetime(df['hourly_timestamp'])
    df = df.sort_values(['station_name', 'hourly_timestamp']).reset_index(drop=True)

    # 2. Build feature set
    print("Reconstructing full feature space for held-out 2023 evaluation...")
    df['hour'] = df['hourly_timestamp'].dt.hour
    df['dayofweek'] = df['hourly_timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['month'] = df['hourly_timestamp'].dt.month

    df = df.merge(df_clusters[['station_name', 'cluster_id', 'cluster_name']], on='station_name', how='left')

    grouped = df.groupby('station_name')['energy_kwh']
    df['lag_1'] = grouped.shift(1)
    df['lag_2'] = grouped.shift(2)
    df['lag_3'] = grouped.shift(3)
    df['lag_4'] = grouped.shift(4)
    df['lag_24'] = grouped.shift(24)
    df['lag_168'] = grouped.shift(168)

    df['rolling_mean_24'] = grouped.transform(lambda x: x.shift(1).rolling(24, min_periods=6).mean())
    df['rolling_max_24'] = grouped.transform(lambda x: x.shift(1).rolling(24, min_periods=6).max())
    df['rolling_mean_168'] = grouped.transform(lambda x: x.shift(1).rolling(168, min_periods=24).mean())

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        'hour', 'dayofweek', 'is_weekend', 'month', 'cluster_id',
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_24', 'lag_168',
        'rolling_mean_24', 'rolling_max_24', 'rolling_mean_168'
    ]
    target_col = 'energy_kwh'

    # 3. Train on 2021-2022, Evaluate strictly on Held-Out 2023
    train_mask = df['hourly_timestamp'] < "2023-01-01 00:00:00"
    test_mask = df['hourly_timestamp'] >= "2023-01-01 00:00:00"

    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target_col]
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, target_col]

    print(f"Training on historical window (2021-2022): {X_train.shape[0]:,} rows")
    print(f"Testing on future held-out year (2023):      {X_test.shape[0]:,} rows")

    model = HistGradientBoostingRegressor(
        loss='poisson',
        max_iter=150,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        categorical_features=['cluster_id', 'is_weekend'],
        random_state=42
    )
    model.fit(X_train, y_train)

    test_df = df.loc[test_mask, ['station_name', 'hourly_timestamp', 'cluster_name', 'hour']].copy()
    test_df['actual'] = y_test.values
    test_df['predicted'] = np.maximum(model.predict(X_test), 0.0)
    test_df['residual'] = test_df['actual'] - test_df['predicted']
    test_df['abs_error'] = np.abs(test_df['residual'])

    overall_wape = calculate_wape(test_df['actual'].values, test_df['predicted'].values)
    overall_mae = mean_absolute_error(test_df['actual'].values, test_df['predicted'].values)
    overall_rmse = np.sqrt(mean_squared_error(test_df['actual'].values, test_df['predicted'].values))
    mean_bias = test_df['residual'].mean()

    print("\n" + "-" * 65)
    print("HELD-OUT 2023 ANNUAL TEST SUMMARY:")
    print("-" * 65)
    print(f"  Overall 2023 WAPE:          {overall_wape:6.2f}%")
    print(f"  Overall 2023 MAE:           {overall_mae:.3f} kWh per hour")
    print(f"  Overall 2023 RMSE:          {overall_rmse:.3f} kWh")
    print(f"  Overall Residual Bias:      {mean_bias:+.4f} kWh (Near zero = Unbiased!)")

    # 4. Cluster-Level Diagnostics
    print("\n" + "-" * 65)
    print("PERFORMANCE BY CLUSTER ARCHETYPE:")
    print("-" * 65)
    cluster_perf = []
    for cname, grp in test_df.groupby('cluster_name'):
        c_wape = calculate_wape(grp['actual'].values, grp['predicted'].values)
        c_mae = grp['abs_error'].mean()
        c_tot = grp['actual'].sum() / 1000.0 # MWh
        cluster_perf.append({
            'cluster_name': cname,
            'wape_pct': round(c_wape, 2),
            'mae_kwh': round(c_mae, 3),
            'total_mwh_2023': round(c_tot, 2),
            'station_count': grp['station_name'].nunique()
        })
        print(f"  {cname:<30} | WAPE: {c_wape:6.2f}% | MAE: {c_mae:.3f} kWh | 2023 Volume: {c_tot:6.1f} MWh")

    # 5. Station-by-Station Rankings
    station_perf = []
    for sname, grp in test_df.groupby('station_name'):
        s_wape = calculate_wape(grp['actual'].values, grp['predicted'].values)
        s_mae = grp['abs_error'].mean()
        s_rmse = np.sqrt(mean_squared_error(grp['actual'].values, grp['predicted'].values))
        s_tot = grp['actual'].sum()
        s_cluster = grp['cluster_name'].iloc[0]
        station_perf.append({
            'station_name': sname,
            'cluster_name': s_cluster,
            'wape_pct': round(s_wape, 2),
            'mae_kwh': round(s_mae, 3),
            'rmse_kwh': round(s_rmse, 3),
            'total_kwh_2023': round(s_tot, 1)
        })

    station_df = pd.DataFrame(station_perf).sort_values('wape_pct', ascending=True)
    os.makedirs("reports", exist_ok=True)
    station_df.to_csv(OUTPUT_STATION_CSV, index=False)
    print(f"\n[SUCCESS] Saved 35-station scorecard to: {OUTPUT_STATION_CSV}")

    print("\nTop 5 Most Predictable Stations (Lowest WAPE):")
    for _, row in station_df.head(5).iterrows():
        print(f"  {row['station_name']:<32} | WAPE: {row['wape_pct']:5.1f}% | Cluster: {row['cluster_name']}")

    print("\nTop 5 Hardest Stations to Predict (Highest WAPE):")
    for _, row in station_df.tail(5).iterrows():
        print(f"  {row['station_name']:<32} | WAPE: {row['wape_pct']:5.1f}% | Cluster: {row['cluster_name']}")

    # 6. Generate Figures
    os.makedirs("reports/figures", exist_ok=True)

    # Chart 07: Residual Diagnostics (Distribution & Hourly Bias)
    print(f"\nGenerating Residual Diagnostics plot: {FIG_RESIDUALS_PATH}...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Subplot A: Residual Distribution (Error Histogram)
    residuals_sample = test_df['residual']
    # Filter extreme outliers for clear visualization
    res_clipped = residuals_sample[residuals_sample.between(-25, 25)]
    axes[0].hist(res_clipped, bins=60, color='#1f77b4', edgecolor='black', alpha=0.75, density=True)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5, label=f'Zero Bias (Mean: {mean_bias:+.3f})')
    axes[0].set_title('Residual Error Distribution (Actual - Predicted)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Residual Error (kWh)', fontsize=10)
    axes[0].set_ylabel('Probability Density', fontsize=10)
    axes[0].legend(loc='upper left', frameon=True)

    # Subplot B: Mean Bias by Hour of Day
    hourly_bias = test_df.groupby('hour')['residual'].mean()
    axes[1].plot(hourly_bias.index, hourly_bias.values, marker='o', color='#2ca02c', linewidth=2.0, label='Mean Hourly Residual')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.2, label='Perfect Calibration (0.0)')
    axes[1].set_title('Hourly Calibration Bias (Are Certain Hours Over/Under-Predicted?)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Hour of the Day (0:00 - 23:00)', fontsize=10)
    axes[1].set_ylabel('Mean Residual (kWh)', fontsize=10)
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].legend(loc='upper right', frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_RESIDUALS_PATH)
    plt.close(fig)
    print(f" -> Saved: {FIG_RESIDUALS_PATH}")

    # Chart 08: Station Error Rankings Bar Chart
    print(f"Generating Station Rankings plot: {FIG_RANKINGS_PATH}...")
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    y_pos = range(len(station_df))
    # Color bars by cluster
    color_map = {
        "High-Traffic Commuter Hub": '#d62728',
        "Afternoon / Evening Hub": '#9467bd',
        "Neighborhood / Community Port": '#1f77b4'
    }
    bar_colors = [color_map.get(c, 'gray') for c in station_df['cluster_name']]

    ax.barh(y_pos, station_df['wape_pct'], color=bar_colors, edgecolor='black', linewidth=0.6, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(station_df['station_name'], fontsize=8)
    ax.set_xlabel('WAPE (%) on 2023 Held-Out Test Set', fontsize=10, fontweight='bold')
    ax.set_title('All 35 Stations Ranked by Forecast Error (WAPE %)', fontsize=12, fontweight='bold', pad=12)

    # Custom legend for clusters
    handles = [plt.Rectangle((0,0),1,1, color=col, ec="black") for col in color_map.values()]
    ax.legend(handles, color_map.keys(), loc='lower right', frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_RANKINGS_PATH)
    plt.close(fig)
    print(f" -> Saved: {FIG_RANKINGS_PATH}")
    print("=" * 65)

if __name__ == "__main__":
    run_evaluation()
