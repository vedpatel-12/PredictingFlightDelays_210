import pandas as pd
import numpy as np
import time

# Change this if your raw CSV has a different filename
RAW_CSV = "flights_sample_3m.csv"
OUTPUT_CSV = "cleaned_flights.csv"

print(f"Loading {RAW_CSV}...")
t0 = time.time()
df = pd.read_csv(RAW_CSV, low_memory=False)
print(f"Loaded in {time.time()-t0:.1f}s — shape: {df.shape}")

# Inspect the data
print("\n--- Column names ---")
print(df.columns.tolist())
print("\n--- Data types ---")
print(df.dtypes)
print("\n--- Missing values per column ---")
print(df.isnull().sum())
print(f"\n--- Duplicate rows: {df.duplicated().sum():,} ---")

# Standardize column names to uppercase
df.columns = df.columns.str.upper().str.strip()

df = df.drop_duplicates()
print(f"\nAfter dropping duplicates: {len(df):,} rows")

# Parse the flight date
df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

# Extract date features from FL_DATE
df["YEAR"]        = df["FL_DATE"].dt.year
df["MONTH"]       = df["FL_DATE"].dt.month
df["DAY"]         = df["FL_DATE"].dt.day
df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek  # 0=Monday, 6=Sunday
df["IS_WEEKEND"]  = (df["DAY_OF_WEEK"] >= 5).astype(int)

# CRS_DEP_TIME is in HHMM format (e.g. 1430 = 2:30 PM), extract just the hour
df["CRS_DEP_TIME"]       = df["CRS_DEP_TIME"].fillna(0).astype(int)
df["SCHEDULED_DEP_HOUR"] = df["CRS_DEP_TIME"] // 100

# Group departure hours into 5 time blocks
def get_time_block(hour):
    if hour < 6:    return 0  # overnight
    elif hour < 12: return 1  # morning
    elif hour < 17: return 2  # afternoon
    elif hour < 21: return 3  # evening
    else:           return 4  # night

df["DEP_TIME_BLOCK"] = df["SCHEDULED_DEP_HOUR"].apply(get_time_block)

# Create the target: 1 if delayed more than 15 min or cancelled, else 0
# The 15-minute threshold is the FAA official definition of a delay
df["DELAYED"] = 0
if "DEP_DELAY" in df.columns:
    df.loc[df["DEP_DELAY"] > 15, "DELAYED"] = 1
if "CANCELLED" in df.columns:
    df.loc[df["CANCELLED"] == 1.0, "DELAYED"] = 1

counts = df["DELAYED"].value_counts()
total  = len(df)
print(f"\n--- Target class balance ---")
print(f"  Not delayed (0): {counts.get(0,0):>10,}  ({counts.get(0,0)/total*100:.1f}%)")
print(f"  Delayed     (1): {counts.get(1,0):>10,}  ({counts.get(1,0)/total*100:.1f}%)")

# Drop leaky columns — these are only known after the flight lands
# Using them as features would let the model cheat
LEAKY_COLS = [
    "DEP_TIME", "DEP_DELAY",
    "ARR_TIME", "ARR_DELAY",
    "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
    "CANCELLED", "CANCELLATION_CODE", "DIVERTED",
    "ELAPSED_TIME", "AIR_TIME",
    "DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS", "DELAY_DUE_SECURITY", "DELAY_DUE_LATE_AIRCRAFT",
]

# Drop redundant columns that add no useful information
REDUNDANT_COLS = [
    "AIRLINE", "AIRLINE_DOT", "DOT_CODE",
    "FL_NUMBER", "ORIGIN_CITY", "DEST_CITY",
]

cols_to_drop = [c for c in LEAKY_COLS + REDUNDANT_COLS if c in df.columns]
df = df.drop(columns=cols_to_drop)
print(f"\nDropped {len(cols_to_drop)} columns.")
print(f"Remaining columns: {df.columns.tolist()}")

# Drop rows with missing target
before = len(df)
df = df.dropna(subset=["DELAYED"])
print(f"\nDropped {before - len(df):,} rows with missing target.")

# Fill missing numeric values with the column median
numeric_cols = df.select_dtypes(include="number").columns.tolist()
for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Fill missing text values with UNKNOWN
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna("UNKNOWN")

# Store FL_DATE as a plain string for the weather join later
df["FL_DATE"] = df["FL_DATE"].dt.strftime("%Y-%m-%d")

print(f"\n--- Final cleaned dataset ---")
print(f"Shape   : {df.shape}")
print(f"Columns : {df.columns.tolist()}")
print(f"Missing : {df.isnull().sum().sum()} total")

print(f"\nSaving to {OUTPUT_CSV}...")
df.to_csv(OUTPUT_CSV, index=False)
print(f"Done! Saved {len(df):,} rows to {OUTPUT_CSV}")
