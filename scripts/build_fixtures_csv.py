import pandas as pd

INPUT_FILE = "data/epl-2025-GMTStandardTime.csv"
OUTPUT_FILE = "data/fixtures_gw24_38.csv"

print("\nBuilding Fixtures CSV (GW24–38)")
print("=" * 50)

# Load the CSV
df = pd.read_csv(INPUT_FILE)
print("Loaded file successfully.")
print("Total matches:", len(df))

# Keep only useful columns
df = df[["Round Number", "Date", "Home Team", "Away Team"]]

# Rename columns to match your system
df.columns = ["Gameweek", "Date", "HomeTeam", "AwayTeam"]

# Convert dates properly
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

# Filter only GW24–GW38
df = df[(df["Gameweek"] >= 24) & (df["Gameweek"] <= 38)]
print("Filtered matches:", len(df))

# Save clean fixtures file
df.to_csv(OUTPUT_FILE, index=False)

print("\nDONE")
print("Saved fixtures here:", OUTPUT_FILE)

print("\nSample output:")
print(df.head())
