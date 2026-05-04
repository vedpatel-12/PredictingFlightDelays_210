"""
fetch_weather.py
----------------
Step 2 of the pipeline. Reads cleaned_flights.csv to find all unique ORIGIN
airports and the date range in the dataset, then fetches daily weather data
from the Open-Meteo Archive API (free, no API key needed).

Results are cached to weather_cache.csv so you only call the API once.
Re-running this file will skip airports that are already cached.

Run after clean_data.py:
    python3 fetch_weather.py
"""

import pandas as pd
import requests
import time
import os

# ── Configuration ──────────────────────────────────────────────────────────────
AIRPORTS_CSV   = "airports.csv"        # lat/lon lookup table
FLIGHTS_CSV    = "cleaned_flights.csv" # produced by clean_data.py
WEATHER_CACHE  = "weather_cache.csv"   # output — one row per airport per day

# The four daily weather variables we want from Open-Meteo.
# These are good predictors of flight delays:
#   temperature_2m_max  — extreme heat can ground or slow aircraft
#   temperature_2m_min  — extreme cold causes ice / de-icing delays
#   precipitation_sum   — rain / snow slows ground ops and reduces visibility
#   wind_speed_10m_max  — high winds can exceed safe crosswind limits
WEATHER_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

# Open-Meteo Archive API endpoint (completely free, no key required)
API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_airport_weather(iata, lat, lon, start_date, end_date):
    """
    Call the Open-Meteo API for one airport for a date range.
    Returns a DataFrame with one row per day and a column per weather variable.
    """
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,   # format: "YYYY-MM-DD"
        "end_date":   end_date,
        "daily":      ",".join(WEATHER_VARS),
        "timezone":   "auto",       # use the local timezone at that lat/lon
    }

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()  # raises an exception if the API returned an error

    data = response.json()

    # The API returns {"daily": {"time": [...], "temp_max": [...], ...}}
    # Convert to a flat DataFrame with one row per day.
    weather_df = pd.DataFrame(data["daily"])
    weather_df.rename(columns={"time": "FL_DATE"}, inplace=True)
    weather_df["ORIGIN"] = iata  # tag with the airport code so we can join later
    return weather_df


# ── Load airport lat/lon lookup ────────────────────────────────────────────────
print(f"Loading airport lookup from {AIRPORTS_CSV}...")
airports = pd.read_csv(AIRPORTS_CSV)
airports = airports.drop_duplicates(subset="IATA", keep="first")
airports = airports.set_index("IATA")  # index by airport code for fast lookup
print(f"  {len(airports)} airports in lookup table.")

# ── Load cleaned flights (only the columns we need) ───────────────────────────
# Reading only ORIGIN and FL_DATE keeps memory use low on the 3M-row file.
print(f"\nReading unique airports and date range from {FLIGHTS_CSV}...")
flights = pd.read_csv(FLIGHTS_CSV, usecols=["ORIGIN", "FL_DATE"])
flights["FL_DATE"] = pd.to_datetime(flights["FL_DATE"])

unique_airports = sorted(flights["ORIGIN"].unique())
start_date = flights["FL_DATE"].min().strftime("%Y-%m-%d")
end_date   = flights["FL_DATE"].max().strftime("%Y-%m-%d")

print(f"  Unique ORIGIN airports : {len(unique_airports)}")
print(f"  Date range             : {start_date} → {end_date}")

# ── Load existing cache (if any) ───────────────────────────────────────────────
# If weather_cache.csv already exists, we only fetch airports not yet in it.
# This makes re-runs fast — you can safely interrupt and resume.
if os.path.exists(WEATHER_CACHE):
    print(f"\nFound existing cache: {WEATHER_CACHE}")
    cached_df      = pd.read_csv(WEATHER_CACHE)
    already_cached = set(cached_df["ORIGIN"].unique())
    all_weather    = [cached_df]
    print(f"  Already cached: {len(already_cached)} airports — will skip these.")
else:
    cached_df      = pd.DataFrame()
    already_cached = set()
    all_weather    = []
    print("\nNo existing cache found — fetching all airports from scratch.")

# ── Fetch weather for each airport ────────────────────────────────────────────
missing_airports = []  # airports not found in our lookup table

for i, iata in enumerate(unique_airports, 1):
    # Skip if already cached
    if iata in already_cached:
        print(f"  [{i:>3}/{len(unique_airports)}] {iata} — skipped (cached)")
        continue

    # Skip if not in our airport lookup table
    if iata not in airports.index:
        print(f"  [{i:>3}/{len(unique_airports)}] {iata} — skipped (not in lookup)")
        missing_airports.append(iata)
        continue

    lat = airports.loc[iata, "LATITUDE"]
    lon = airports.loc[iata, "LONGITUDE"]

    print(f"  [{i:>3}/{len(unique_airports)}] {iata} ({lat:.2f}, {lon:.2f}) fetching {start_date} → {end_date}...", end=" ")

    try:
        weather_df = fetch_airport_weather(iata, lat, lon, start_date, end_date)
        all_weather.append(weather_df)
        print(f"{len(weather_df)} days")

        # Be polite to the API — small pause between requests
        time.sleep(0.4)

    except requests.exceptions.RequestException as e:
        print(f"ERROR: {e}")

# ── Combine all fetched weather into one DataFrame and save ───────────────────
if all_weather:
    combined = pd.concat(all_weather, ignore_index=True)

    # Ensure FL_DATE is stored as a consistent date string
    combined["FL_DATE"] = pd.to_datetime(combined["FL_DATE"]).dt.strftime("%Y-%m-%d")

    combined.to_csv(WEATHER_CACHE, index=False)
    print(f"\nWeather cache saved to {WEATHER_CACHE}")
    print(f"Shape: {combined.shape}")
    print(combined.head())
else:
    print("\nNo weather data was fetched.")

if missing_airports:
    print(f"\nAirports not found in lookup ({len(missing_airports)}): {missing_airports}")
    print("These flights will have NaN weather features after the merge.")
