import pandas as pd

# Normalise team names so fixtures + history match
TEAM_NAME_MAP = {
    "Spurs": "Tottenham Hotspur",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",

}


def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)

def load_fixtures(path):
    return pd.read_csv(path)


def build_played_set(history_df: pd.DataFrame) -> set:
    """
    Build a fast lookup set of played matches:
    (HomeTeam, AwayTeam)
    """
    played = set()

    for _, row in history_df.iterrows():
        home = normalize_team(row["HomeTeam"])
        away = normalize_team(row["AwayTeam"])
        played.add((home, away))

    return played


def get_next_matchweek(fixtures_df, history_df):
    fixtures_df = fixtures_df.copy()
    history_df = history_df.copy()

    fixtures_df["Date"] = pd.to_datetime(fixtures_df["Date"])
    history_df["Date"] = pd.to_datetime(history_df["Date"])

    last_played_date = history_df["Date"].max()

    future_fixtures = fixtures_df[
        fixtures_df["Date"] > last_played_date
    ]

    if future_fixtures.empty:
        raise ValueError("No future fixtures found")

    return future_fixtures["Gameweek"].min()
