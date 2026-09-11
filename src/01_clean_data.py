"""
01_clean_data.py
----------------
Phase 1: Ingestion, Filtering, and Regular Hourly Time-Series Resampling.

Steps:
1. Load raw transaction records from data/raw/boulder_ev_charging.csv.
2. Filter to 2021-2023 window (consistent post-lockdown adoption era).
3. Clean corrupted rows (zero/negative energy, invalid dates).
4. Identify active stations with consistent history.
5. Create a complete hourly clock per station and fill quiet hours with 0.0 kWh.
6. Export cleaned dataset to data/processed/hourly_station_demand.csv.
"""

import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/raw/boulder_ev_charging.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hourly_station_demand.csv")

def run_data_cleaning():
    print("=" * 60)
    print("STEP 1: INGESTION & DATA CLEANING PIPELINE")
    print("=" * 60)

    # 1. Load raw dataset
    print(f"Loading raw transactions from: {RAW_DATA_PATH}")
    cols_to_use = [
        'Station_Name', 
        'Start_Date___Time', 
        'End_Date___Time', 
        'Energy__kWh_', 
        'Total_Duration__hh_mm_ss_',
        'Port_Type'
    ]
    df = pd.read_csv(RAW_DATA_PATH, usecols=cols_to_use)
    print(f"Total raw rows loaded: {len(df):,}")

    # 2. Parse timestamps and handle errors
    print("Parsing timestamps...")
    df['start_time'] = pd.to_datetime(df['Start_Date___Time'], errors='coerce')
    df = df.dropna(subset=['start_time'])
    
    # 3. Filter to 2021-2023 window
    print("Filtering to 2021-01-01 through 2023-12-31...")
    start_date = pd.Timestamp("2021-01-01 00:00:00")
    end_date = pd.Timestamp("2023-12-31 23:59:59")
    df = df[(df['start_time'] >= start_date) & (df['start_time'] <= end_date)].copy()
    print(f"Rows in 2021-2023 window: {len(df):,}")

    # 4. Clean energy values (drop non-positive or corrupted readings)
    initial_window_rows = len(df)
    df = df[df['Energy__kWh_'] > 0].copy()
    dropped_zeroes = initial_window_rows - len(df)
    print(f"Dropped {dropped_zeroes:,} invalid/zero-energy sessions. Valid sessions: {len(df):,}")

    # 5. Clean station names and filter consistent stations
    df['station_name'] = df['Station_Name'].astype(str).str.strip()
    
    # Filter stations that have at least 150 charging sessions across the 3 years (removes transient/test ports)
    station_counts = df['station_name'].value_counts()
    active_stations = station_counts[station_counts >= 150].index.tolist()
    df = df[df['station_name'].isin(active_stations)].copy()
    print(f"Identified {len(active_stations)} active stations with consistent charging history.")

    # 6. Hourly Aggregation (Resampling)
    print("Constructing regular continuous hourly grid per station...")
    # Floor timestamps to the starting hour (e.g. 09:14 -> 09:00)
    df['hourly_timestamp'] = df['start_time'].dt.floor('h')

    # Sum energy per station per hour
    hourly_agg = (
        df.groupby(['station_name', 'hourly_timestamp'])['Energy__kWh_']
        .sum()
        .reset_index()
        .rename(columns={'Energy__kWh_': 'energy_kwh'})
    )

    # 7. Create full continuous grid for every station from start_date to end_date
    full_hours = pd.date_range(start="2021-01-01 00:00:00", end="2023-12-31 23:00:00", freq='1h')
    
    # MultiIndex grid: all stations x all hours
    grid_index = pd.MultiIndex.from_product(
        [active_stations, full_hours], 
        names=['station_name', 'hourly_timestamp']
    )
    
    # Reindex and fill quiet hours with 0.0
    full_grid = (
        hourly_agg.set_index(['station_name', 'hourly_timestamp'])
        .reindex(grid_index, fill_value=0.0)
        .reset_index()
    )

    # Round energy values to 3 decimal places for cleanliness
    full_grid['energy_kwh'] = full_grid['energy_kwh'].round(3)

    # 8. Export to data/processed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_grid.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[SUCCESS] Clean hourly demand dataset saved to: {OUTPUT_FILE}")
    print(f"Total continuous hourly rows: {len(full_grid):,}")
    print(f"Hourly time range: {full_hours[0]} to {full_hours[-1]}")
    print(f"Total stations tracked: {len(active_stations)}")
    print(f"Total electricity accounted for: {full_grid['energy_kwh'].sum():,.2f} kWh")

    # Sample preview
    print("\nSample Preview (First 5 rows):")
    print(full_grid.head())
    print("=" * 60)

if __name__ == "__main__":
    run_data_cleaning()
