import pandas as pd


FEATURE_COLUMNS = [
    "HomeForm5",
    "AwayForm5",
    "HomeGoalsAvg",
    "AwayGoalsAvg",
    "HomeConcededAvg",
    "AwayConcededAvg",
    "FormCloseness",   
    "GoalsCloseness"
]

TEAM_NAME_MAP = {
    "Spurs": "Tottenham",
    "Man City": "Man City",
    "Man Utd": "Man United",
    "Nott'm Forest": "Nott'm Forest",
    "Wolves": "Wolves",
}


def build_prediction_features(df, home, away):
    """
    Build prediction feature vector using last 5 games of each team.
    Compatible with E0.csv column names.
    """

    # Normalize team names
    home = TEAM_NAME_MAP.get(home, home)
    away = TEAM_NAME_MAP.get(away, away)

    # Last 5 home matches
    home_matches = df[df["HomeTeam"] == home].tail(5)

    # Last 5 away matches
    away_matches = df[df["AwayTeam"] == away].tail(5)

    # If not enough history → still allow prediction
    if len(home_matches) < 3 or len(away_matches) < 3:
        return None

    # Goals scored
    home_goals_avg = home_matches["FullTimeHomeTeamGoals"].mean()
    away_goals_avg = away_matches["FullTimeAwayTeamGoals"].mean()
    
    # Goals conceded
    home_conceded_avg = home_matches["FullTimeAwayTeamGoals"].mean()
    away_conceded_avg = away_matches["FullTimeHomeTeamGoals"].mean()

    # Form points
    home_form = home_matches["HomeTeamPoints"].sum()
    away_form = away_matches["AwayTeamPoints"].sum()

    # Closeness 
    form_closeness = abs(home_form - away_form)
    goals_closeness = abs(home_goals_avg - away_goals_avg)

    return pd.DataFrame([{
        "HomeForm5": home_form,
        "AwayForm5": away_form,
        "HomeGoalsAvg": home_goals_avg,
        "AwayGoalsAvg": away_goals_avg,
        "HomeConcededAvg": home_conceded_avg,
        "AwayConcededAvg": away_conceded_avg,
        "FormCloseness": form_closeness,
        "GoalsCloseness": goals_closeness,
    }])[FEATURE_COLUMNS]