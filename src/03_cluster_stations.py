"""
03_cluster_stations.py
----------------------
Phase 3: Station Profiling & Behavioral Clustering.

1. Loads the cleaned hourly dataset (data/processed/hourly_station_demand.csv).
2. Extracts a 4-dimensional behavioral fingerprint for each of the 35 stations:
   - daily_volume_kwh: Mean daily electricity throughput (kWh/day).
   - weekday_share: Proportion of energy consumed on weekdays (0.0 - 1.0; 0.714 is natural baseline).
   - peak_hour: Hour of the day (0-23) with the highest average draw.
   - load_factor: Demand consistency (mean load / peak load).
3. Standardizes features with StandardScaler.
4. Performs K-Means clustering (K=3) to group stations into intuitive archetypes.
5. Saves station assignments to data/processed/station_clusters.csv.
6. Exports high-res visualization to reports/figures/05_station_clusters.png.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

INPUT_FILE = "data/processed/hourly_station_demand.csv"
OUTPUT_CLUSTERS_FILE = "data/processed/station_clusters.csv"
OUTPUT_FIG_PATH = "reports/figures/05_station_clusters.png"

def run_clustering():
    print("=" * 60)
    print("STEP 3: STATION BEHAVIORAL PROFILING & K-MEANS CLUSTERING")
    print("=" * 60)

    # 1. Load data
    print(f"Loading hourly demand dataset from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df['hourly_timestamp'] = pd.to_datetime(df['hourly_timestamp'])

    # 2. Derive time attributes
    df['hour'] = df['hourly_timestamp'].dt.hour
    df['is_weekend'] = df['hourly_timestamp'].dt.dayofweek.isin([5, 6])

    print("Extracting behavioral fingerprints for all 35 stations...")

    # Feature 1: Average daily volume (kWh/day)
    daily_volume = (df.groupby('station_name')['energy_kwh'].mean() * 24.0).round(2)

    # Feature 2: Weekday Share of Total Energy (avoids division-by-zero ratio outliers)
    total_kwh = df.groupby('station_name')['energy_kwh'].sum()
    weekday_kwh = df[~df['is_weekend']].groupby('station_name')['energy_kwh'].sum()
    weekday_share = (weekday_kwh / total_kwh).round(3)

    # Feature 3: Peak Hour (the hour 0-23 with highest mean demand)
    hourly_means = df.groupby(['station_name', 'hour'])['energy_kwh'].mean().unstack()
    peak_hour = hourly_means.idxmax(axis=1)
    peak_demand = hourly_means.max(axis=1)

    # Feature 4: Load Factor (mean demand / peak hourly demand)
    mean_demand = df.groupby('station_name')['energy_kwh'].mean()
    load_factor = (mean_demand / peak_demand).round(3)

    # Assemble fingerprint dataframe
    profiles = pd.DataFrame({
        'daily_volume_kwh': daily_volume,
        'weekday_share': weekday_share,
        'peak_hour': peak_hour,
        'load_factor': load_factor
    }).reset_index()

    feature_cols = ['daily_volume_kwh', 'weekday_share', 'peak_hour', 'load_factor']
    X = profiles[feature_cols].values

    # 3. Standardize features
    print("Normalizing multi-dimensional features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. K-Means clustering (K=3)
    K = 3
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    profiles['cluster_id'] = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, profiles['cluster_id'])
    print(f"K-Means fitted with K={K} (Silhouette Score: {score:.3f})")

    # 5. Label clusters dynamically based on physical characteristics
    cluster_means = profiles.groupby('cluster_id')[feature_cols].mean()
    print("\n" + "-" * 60)
    print("CLUSTER CENTROIDS (Average Stats per Group):")
    print("-" * 60)
    print(cluster_means)

    cluster_names = {}
    for cid in range(K):
        vol = cluster_means.loc[cid, 'daily_volume_kwh']
        pk = cluster_means.loc[cid, 'peak_hour']
        if vol >= 40.0:
            cluster_names[cid] = "High-Traffic Commuter Hub"
        elif pk >= 14.0:
            cluster_names[cid] = "Afternoon / Evening Hub"
        else:
            cluster_names[cid] = "Neighborhood / Community Port"

    profiles['cluster_name'] = profiles['cluster_id'].map(cluster_names)

    print("\n" + "-" * 60)
    print("STATION CLUSTER ASSIGNMENT SUMMARY:")
    print("-" * 60)
    for c_id, count in profiles['cluster_id'].value_counts().items():
        name = cluster_names[c_id]
        print(f" Cluster {c_id} [{name}]: {count} stations")

    # 6. Export to CSV
    os.makedirs("data/processed", exist_ok=True)
    profiles.to_csv(OUTPUT_CLUSTERS_FILE, index=False)
    print(f"\n[SUCCESS] Station clusters exported to: {OUTPUT_CLUSTERS_FILE}")

    # 7. Generate Visualizations
    print(f"Generating cluster visualization: {OUTPUT_FIG_PATH}...")
    os.makedirs("reports/figures", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    colors = {
        "High-Traffic Commuter Hub": '#d62728',     # Red
        "Afternoon / Evening Hub": '#9467bd',       # Purple
        "Neighborhood / Community Port": '#1f77b4'  # Blue
    }

    # Panel A: Daily Volume vs Peak Hour
    for cname, col in colors.items():
        sub = profiles[profiles['cluster_name'] == cname]
        axes[0].scatter(
            sub['peak_hour'],
            sub['daily_volume_kwh'],
            label=f"{cname} (n={len(sub)})",
            color=col,
            s=80,
            alpha=0.85,
            edgecolors='black',
            linewidths=0.8
        )

    axes[0].set_title('Station Clusters: Peak Hour vs. Daily Volume', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Peak Charging Hour of Day (0:00 - 23:00)', fontsize=11)
    axes[0].set_ylabel('Average Daily Volume (kWh/day)', fontsize=11)
    axes[0].set_xticks(range(6, 20, 2))
    axes[0].legend(frameon=True, facecolor='white', loc='upper right', fontsize=9)

    # Panel B: Average 24-Hour Diurnal Curves by Cluster
    df_clustered = df.merge(profiles[['station_name', 'cluster_name']], on='station_name')
    cluster_hourly = df_clustered.groupby(['cluster_name', 'hour'])['energy_kwh'].mean().unstack(level=0)

    for cname, col in colors.items():
        if cname in cluster_hourly.columns:
            axes[1].plot(
                cluster_hourly.index,
                cluster_hourly[cname],
                marker='o',
                linewidth=2.4,
                color=col,
                label=cname
            )

    axes[1].set_title('Average 24-Hour Load Curve by Cluster Archetype', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Hour of the Day (0:00 - 23:00)', fontsize=11)
    axes[1].set_ylabel('Average Hourly Energy (kWh)', fontsize=11)
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].legend(frameon=True, facecolor='white', loc='upper left', fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_PATH)
    plt.close(fig)
    print(f" -> Saved: {OUTPUT_FIG_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    run_clustering()
