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
    """
    Find the next gameweek to predict based on completed gameweeks in history.
    Looks at Season 2025-2026 specifically.
    """
    fixtures_df = fixtures_df.copy()
    history_df = history_df.copy()
    
    # Get current season from history
    current_season = history_df[history_df["Season"] == "2025-2026"]
    
    if current_season.empty:
       
        return 24
    
   
    gameweeks = current_season.groupby("MatchWeek").size()
    
    # Find highest gameweek with 10 matches 
    complete_gameweeks = gameweeks[gameweeks == 10].index
    
    if len(complete_gameweeks) == 0:
        return 24  
    
    last_complete = complete_gameweeks.max()
    next_gw = last_complete + 1
    
    print(f"Last complete gameweek: {last_complete}")
    print(f"Next gameweek to predict: {next_gw}")
    
    return next_gw