import pandas as pd

FEATURE_COLUMNS = [
    "HomeTeamShots",
    "AwayTeamShots",
    "HomeTeamShotsOnTarget",
    "AwayTeamShotsOnTarget",
    "HomeTeamCorners",
    "AwayTeamCorners",
    "B365HomeTeam",
    "B365Draw",
    "B365AwayTeam",
    "HomeForm5",
    "AwayForm5",
]


def build_prediction_features(df, home_team, away_team):
    """
    Builds a feature row for prediction.
    If a team has no history, fallback to league averages.
    """

    #  League fallback averages 
    league_avg = df[FEATURE_COLUMNS].mean()

    #  Team history 
    home_matches = df[(df["HomeTeam"] == home_team) | (df["AwayTeam"] == home_team)]
    away_matches = df[(df["HomeTeam"] == away_team) | (df["AwayTeam"] == away_team)]

    #  If missing team history, fallback 
    if len(home_matches) < 5:
        print(f" No history for {home_team}, using league averages")

    if len(away_matches) < 5:
        print(f" No history for {away_team}, using league averages")

    
    features = {
        "HomeTeamShots": home_matches["HomeTeamShots"].mean() if len(home_matches) > 0 else league_avg["HomeTeamShots"],
        "AwayTeamShots": away_matches["AwayTeamShots"].mean() if len(away_matches) > 0 else league_avg["AwayTeamShots"],

        "HomeTeamShotsOnTarget": home_matches["HomeTeamShotsOnTarget"].mean()
        if len(home_matches) > 0 else league_avg["HomeTeamShotsOnTarget"],

        "AwayTeamShotsOnTarget": away_matches["AwayTeamShotsOnTarget"].mean()
        if len(away_matches) > 0 else league_avg["AwayTeamShotsOnTarget"],

        "HomeTeamCorners": home_matches["HomeTeamCorners"].mean()
        if len(home_matches) > 0 else league_avg["HomeTeamCorners"],

        "AwayTeamCorners": away_matches["AwayTeamCorners"].mean()
        if len(away_matches) > 0 else league_avg["AwayTeamCorners"],

        # Odds fallback
        "B365HomeTeam": league_avg["B365HomeTeam"],
        "B365Draw": league_avg["B365Draw"],
        "B365AwayTeam": league_avg["B365AwayTeam"],

        # Form fallback
        "HomeForm5": home_matches["HomeForm5"].iloc[-1] if len(home_matches) > 0 else league_avg["HomeForm5"],
        "AwayForm5": away_matches["AwayForm5"].iloc[-1] if len(away_matches) > 0 else league_avg["AwayForm5"],
    }

    return pd.DataFrame([features])
