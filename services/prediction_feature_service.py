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
    Build a single feature row for prediction.
    Always returns something (uses league averages if team has no history).
    """

    # League-wide averages (fallback)
    league_avg = df[FEATURE_COLUMNS].mean()

    # Match history for each team
    home_matches = df[
        (df["HomeTeam"] == home_team) | (df["AwayTeam"] == home_team)
    ].tail(10)

    away_matches = df[
        (df["HomeTeam"] == away_team) | (df["AwayTeam"] == away_team)
    ].tail(10)

   
    if home_matches.empty:
        home_stats = league_avg
    else:
        home_stats = home_matches[FEATURE_COLUMNS].mean()

    if away_matches.empty:
        away_stats = league_avg
    else:
        away_stats = away_matches[FEATURE_COLUMNS].mean()

    # Build final row
    row = {
        "HomeTeamShots": home_stats["HomeTeamShots"],
        "AwayTeamShots": away_stats["AwayTeamShots"],
        "HomeTeamShotsOnTarget": home_stats["HomeTeamShotsOnTarget"],
        "AwayTeamShotsOnTarget": away_stats["AwayTeamShotsOnTarget"],
        "HomeTeamCorners": home_stats["HomeTeamCorners"],
        "AwayTeamCorners": away_stats["AwayTeamCorners"],
        "B365HomeTeam": home_stats["B365HomeTeam"],
        "B365Draw": home_stats["B365Draw"],
        "B365AwayTeam": away_stats["B365AwayTeam"],
        "HomeForm5": home_stats["HomeForm5"],
        "AwayForm5": away_stats["AwayForm5"],
    }

    return pd.DataFrame([row])
