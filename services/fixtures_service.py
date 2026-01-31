import pandas as pd


# Load Fixtures from CSV
def load_fixtures(path):
    df = pd.read_csv(path)

    # Convert Date column properly
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    return df



# Get Next Matchweek

def get_next_matchweek(fixtures_df):

    if fixtures_df.empty:
        return fixtures_df

    # Find the smallest upcoming gameweek
    next_gw = fixtures_df["Gameweek"].min()

    print(f"Next Gameweek Detected: {next_gw}")

    # Return all matches in that gameweek
    return fixtures_df[fixtures_df["Gameweek"] == next_gw]
