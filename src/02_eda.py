"""
02_eda.py
---------
Phase 2: Exploratory Data Analysis (EDA) & Visualization Suite.

Analyzes the regularized hourly dataset to uncover:
1. Diurnal Cycle (24-hour daily curve).
2. Day-of-Week & Weekday vs. Weekend behavioral contrast.
3. Macro Trends: Monthly aggregate consumption & year-over-year adoption growth (2021-2023).
4. Cross-Station Heterogeneity: Disparity between top-loaded vs quiet charging hubs.

Outputs high-resolution charts to `reports/figures/`.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "data/processed/hourly_station_demand.csv"
OUTPUT_FIG_DIR = "reports/figures"

def run_eda():
    print("=" * 60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA) PIPELINE")
    print("=" * 60)

    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

    # 1. Load regularized dataset
    print(f"Loading cleaned hourly dataset from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df['hourly_timestamp'] = pd.to_datetime(df['hourly_timestamp'])
    
    # 2. Extract temporal dimensions
    print("Deriving time features (hour, day of week, month, year, weekend)...")
    df['hour'] = df['hourly_timestamp'].dt.hour
    df['dayofweek'] = df['hourly_timestamp'].dt.dayofweek # 0 = Mon, 6 = Sun
    df['day_name'] = df['hourly_timestamp'].dt.day_name()
    df['month'] = df['hourly_timestamp'].dt.month
    df['year'] = df['hourly_timestamp'].dt.year
    df['year_month'] = df['hourly_timestamp'].dt.to_period('M')
    df['is_weekend'] = df['dayofweek'].isin([5, 6])

    # Print summary statistics
    print("\n" + "-" * 40)
    print("DESCRIPTIVE SUMMARY STATISTICS")
    print("-" * 40)
    print(f"Total rows analyzed: {len(df):,}")
    print(f"Number of active stations: {df['station_name'].nunique()}")
    print(f"Total kWh delivered (2021-2023): {df['energy_kwh'].sum():,.2f} kWh")
    print(f"Overall average hourly load per station: {df['energy_kwh'].mean():.3f} kWh")
    print(f"Overall max single-hour station spike: {df['energy_kwh'].max():.2f} kWh")
    zero_ratio = (df['energy_kwh'] == 0).mean() * 100
    print(f"Zero-demand (quiet) hours percentage: {zero_ratio:.1f}%")

    # Set clean Matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

    # -------------------------------------------------------------
    # CHART 1: Average Diurnal Cycle (24-Hour Load Shape)
    # -------------------------------------------------------------
    print("\nGenerating Chart 1: 24-Hour Diurnal Profile...")
    hourly_stats = df.groupby('hour')['energy_kwh'].agg(['mean', 'median', lambda x: np.percentile(x, 75)])
    hourly_stats.columns = ['mean', 'median', 'p75']

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(hourly_stats.index, hourly_stats['mean'], color='#1f77b4', linewidth=2.5, marker='o', label='Mean Demand (kWh)')
    ax.fill_between(hourly_stats.index, 0, hourly_stats['mean'], color='#1f77b4', alpha=0.15)
    ax.plot(hourly_stats.index, hourly_stats['p75'], color='#ff7f0e', linestyle='--', linewidth=1.8, label='75th Percentile')
    
    # Highlight peak charging window
    peak_hour = hourly_stats['mean'].idxmax()
    ax.axvline(x=peak_hour, color='#d62728', linestyle=':', alpha=0.8)
    ax.annotate(f'Peak Hour: {peak_hour}:00\n({hourly_stats["mean"].max():.2f} kWh avg)',
                xy=(peak_hour, hourly_stats['mean'].max()),
                xytext=(peak_hour + 1.2, hourly_stats['mean'].max() * 0.92),
                arrowprops=dict(facecolor='#d62728', arrowstyle='->', lw=1.5),
                fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffecec', edgecolor='#d62728', alpha=0.8))

    ax.set_title('Boulder EV Network: Average 24-Hour Diurnal Demand Curve (2021-2023)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Hour of the Day (0:00 - 23:00)', fontsize=11)
    ax.set_ylabel('Average Energy Demand per Station (kWh)', fontsize=11)
    ax.set_xticks(range(0, 24))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(frameon=True, facecolor='white', loc='upper left')
    fig.tight_layout()
    chart1_path = os.path.join(OUTPUT_FIG_DIR, "01_diurnal_hourly_profile.png")
    fig.savefig(chart1_path)
    plt.close(fig)
    print(f" -> Saved: {chart1_path}")

    # -------------------------------------------------------------
    # CHART 2: Weekday vs. Weekend Demand Profile
    # -------------------------------------------------------------
    print("Generating Chart 2: Weekday vs. Weekend Comparison...")
    weekday_profile = df[~df['is_weekend']].groupby('hour')['energy_kwh'].mean()
    weekend_profile = df[df['is_weekend']].groupby('hour')['energy_kwh'].mean()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(weekday_profile.index, weekday_profile.values, color='#2ca02c', linewidth=2.5, marker='s', label='Weekdays (Mon - Fri)')
    ax.plot(weekend_profile.index, weekend_profile.values, color='#9467bd', linewidth=2.5, marker='^', linestyle='--', label='Weekends (Sat - Sun)')
    ax.fill_between(weekday_profile.index, weekday_profile.values, weekend_profile.values, color='gray', alpha=0.1, label='Commuter Gap')

    ax.set_title('EV Demand Rhythm: Weekday (Commuter) vs. Weekend (Leisure)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Hour of the Day', fontsize=11)
    ax.set_ylabel('Average Energy per Station (kWh)', fontsize=11)
    ax.set_xticks(range(0, 24))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(frameon=True, facecolor='white', loc='upper left')
    fig.tight_layout()
    chart2_path = os.path.join(OUTPUT_FIG_DIR, "02_weekday_vs_weekend.png")
    fig.savefig(chart2_path)
    plt.close(fig)
    print(f" -> Saved: {chart2_path}")

    # -------------------------------------------------------------
    # CHART 3: Macro Monthly Trend & Year-over-Year Growth
    # -------------------------------------------------------------
    print("Generating Chart 3: Monthly Macro Trend (2021-2023)...")
    monthly_total = df.groupby('year_month')['energy_kwh'].sum() / 1000.0 # MWh
    months_str = [str(p) for p in monthly_total.index]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    bars = ax.bar(months_str, monthly_total.values, color='#17becf', edgecolor='#0e7f8a', alpha=0.85)
    
    # Trendline (moving average)
    ma_3 = monthly_total.rolling(window=3, min_periods=1).mean()
    ax.plot(months_str, ma_3.values, color='#d62728', linewidth=2.2, label='3-Month Rolling Trend')

    ax.set_title('Boulder EV Network Monthly Consumption (MWh delivered, 2021-2023)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Year-Month', fontsize=11)
    ax.set_ylabel('Total Electricity Delivered (MWh)', fontsize=11)
    ax.set_xticks(range(len(months_str)))
    ax.set_xticklabels(months_str, rotation=45, ha='right', fontsize=9)
    ax.legend(frameon=True, facecolor='white', loc='upper left')
    fig.tight_layout()
    chart3_path = os.path.join(OUTPUT_FIG_DIR, "03_monthly_trend_growth.png")
    fig.savefig(chart3_path)
    plt.close(fig)
    print(f" -> Saved: {chart3_path}")

    # -------------------------------------------------------------
    # CHART 4: Station Heterogeneity (Top 12 Stations by Load)
    # -------------------------------------------------------------
    print("Generating Chart 4: Station Load Distribution...")
    top_stations = df.groupby('station_name')['energy_kwh'].sum().sort_values(ascending=True).tail(15) / 1000.0 # MWh

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    y_pos = range(len(top_stations))
    ax.barh(y_pos, top_stations.values, color='#3b528b', edgecolor='#212f54', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_stations.index, fontsize=9)
    ax.set_title('Top 15 Highest-Demand Charging Stations (Cumulative MWh, 2021-2023)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Total Electricity Delivered (MWh)', fontsize=11)
    
    # Add value annotations
    for i, v in enumerate(top_stations.values):
        ax.text(v + 0.8, i, f"{v:.1f} MWh", va='center', fontsize=8, color='#111111')

    fig.tight_layout()
    chart4_path = os.path.join(OUTPUT_FIG_DIR, "04_station_demand_disparity.png")
    fig.savefig(chart4_path)
    plt.close(fig)
    print(f" -> Saved: {chart4_path}")

    print("\n" + "=" * 60)
    print(f"[SUCCESS] All 4 EDA charts successfully exported to: {OUTPUT_FIG_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()
