import pandas as pd

FIXTURE_PATH = "data/2024-2025 fixtures.csv"

def load_next_matchweek():
    df = pd.read_csv(FIXTURE_PATH)

    # Only games not played yet
    df_upcoming = df[df["FullTimeResult"].isna()]

    if len(df_upcoming) == 0:
        return None, []

    # Find the next matchweek number
    next_week = df_upcoming["MatchWeek"].min()

    # Get all fixtures in that week
    matchweek_games = df_upcoming[df_upcoming["MatchWeek"] == next_week]

    fixtures = matchweek_games[["HomeTeam", "AwayTeam", "Date"]]

    return next_week, fixtures.to_dict(orient="records")
