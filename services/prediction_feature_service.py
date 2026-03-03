import pandas as pd


FEATURE_COLUMNS = [
    "HomeForm5",
    "AwayForm5",
    "HomeGoalsAvg",
    "AwayGoalsAvg",
    "FormCloseness",
    "GoalsCloseness",
    "HomeDrawRate",
    "AwayDrawRate",
    "HomePosition",
    "AwayPosition",
    "PositionGap",
]

TEAM_NAME_MAP = {
    "Spurs": "Tottenham Hotspur",
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

    # Draw rates 
    home_draw_rate = (home_matches["FullTimeResult"] == 'D').sum() / len(home_matches)
    away_draw_rate = (away_matches["FullTimeResult"] == 'D').sum() / len(away_matches)

    # Position based on recent form ranking across all teams
    # all teams' last 5-game form
    all_teams_home = df.groupby("HomeTeam").tail(5).groupby("HomeTeam")["HomeTeamPoints"].sum()
    all_teams_away = df.groupby("AwayTeam").tail(5).groupby("AwayTeam")["AwayTeamPoints"].sum()

    # Combine into single form ranking
    team_form_dict = {}
    for team in set(list(all_teams_home.index) + list(all_teams_away.index)):
        home_pts = all_teams_home.get(team, 0)
        away_pts = all_teams_away.get(team, 0)
        team_form_dict[team] = home_pts + away_pts

    # Convert to Series and rank
    team_forms = pd.Series(team_form_dict)
    team_positions = team_forms.rank(ascending=False, pct=True)

    home_position = team_positions.get(home, 0.5)
    away_position = team_positions.get(away, 0.5)
    position_gap = abs(home_position - away_position)

    return pd.DataFrame([{
        "HomeForm5": home_form,
        "AwayForm5": away_form,
        "HomeGoalsAvg": home_goals_avg,
        "AwayGoalsAvg": away_goals_avg,
        "FormCloseness": form_closeness,
        "GoalsCloseness": goals_closeness,
        "HomeDrawRate": home_draw_rate,
        "AwayDrawRate": away_draw_rate,
        "HomePosition": home_position,
        "AwayPosition": away_position,
        "PositionGap": position_gap,
    }])[FEATURE_COLUMNS]