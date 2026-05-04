"""
merge_weather.py
----------------
Step 3 of the pipeline. Joins the cleaned flight data with the weather cache
on ORIGIN airport code + FL_DATE, then saves the enriched dataset as
flights_with_weather.csv.

Run after fetch_weather.py:
    python3 merge_weather.py
"""

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
FLIGHTS_CSV = "cleaned_flights.csv"   # produced by clean_data.py
WEATHER_CSV = "weather_cache.csv"     # produced by fetch_weather.py
OUTPUT_CSV  = "flights_with_weather.csv"

# ── Load flights ───────────────────────────────────────────────────────────────
print(f"Loading {FLIGHTS_CSV}...")
flights = pd.read_csv(FLIGHTS_CSV, low_memory=False)
print(f"  Shape: {flights.shape}")

# Normalize FL_DATE to a plain date string (YYYY-MM-DD) so the join key matches.
flights["FL_DATE"] = pd.to_datetime(flights["FL_DATE"]).dt.strftime("%Y-%m-%d")

# ── Load weather cache ─────────────────────────────────────────────────────────
print(f"\nLoading {WEATHER_CSV}...")
weather = pd.read_csv(WEATHER_CSV)
print(f"  Shape: {weather.shape}")
print(f"  Columns: {weather.columns.tolist()}")

# Normalize FL_DATE in weather cache as well
weather["FL_DATE"] = pd.to_datetime(weather["FL_DATE"]).dt.strftime("%Y-%m-%d")

# ── Merge flights + weather ────────────────────────────────────────────────────
# Left join keeps ALL flight rows. Flights at airports not in the weather cache
# will have NaN for the weather columns (handled in model files via fillna).
#
# Join key: ORIGIN airport code + FL_DATE
#   - Each airport has one set of daily weather values per day
#   - Every flight at that airport on that day gets the same weather values
print("\nMerging flights with weather on ORIGIN + FL_DATE...")
merged = flights.merge(weather, on=["ORIGIN", "FL_DATE"], how="left")
print(f"  Merged shape: {merged.shape}")

# ── Report match quality ───────────────────────────────────────────────────────
total   = len(merged)
matched = merged["temperature_2m_max"].notna().sum()
missing = total - matched

print(f"\n  Weather matched : {matched:>10,} rows  ({matched/total*100:.1f}%)")
print(f"  No weather data : {missing:>10,} rows  ({missing/total*100:.1f}%)")

if missing > 0:
    # Show which airports are missing weather — helps diagnose lookup gaps
    missing_airports = (
        merged[merged["temperature_2m_max"].isna()]["ORIGIN"]
        .value_counts()
        .head(10)
    )
    print(f"\n  Top airports with missing weather:")
    print(missing_airports.to_string())

# ── Fill NaN weather values with column medians ────────────────────────────────
# Rather than dropping rows that lack weather data, fill with the median so
# those flights can still be used for training.
weather_cols = ["temperature_2m_max", "temperature_2m_min",
                "precipitation_sum", "wind_speed_10m_max"]
for col in weather_cols:
    if col in merged.columns and merged[col].isnull().any():
        merged[col] = merged[col].fillna(merged[col].median())

# ── Save ───────────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUTPUT_CSV}...")
merged.to_csv(OUTPUT_CSV, index=False)
print(f"Done! {OUTPUT_CSV} saved with shape {merged.shape}")
print(f"\nFinal columns: {merged.columns.tolist()}")
