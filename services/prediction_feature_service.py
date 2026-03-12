import pandas as pd
from services.standings_service import get_current_standings

FEATURE_COLUMNS = [
    "HomeForm5",
    "AwayForm5",
    "HomeGoalsAvg",
    "AwayGoalsAvg",
    "HomeConcededAvg",
    "AwayConcededAvg",
    "FormCloseness",
    "GoalsCloseness",
    "HomeDrawRate",
    "AwayDrawRate",
    "HomePosition",      
    "AwayPosition",      
    "PositionGap",       
    "HomeTablePos",      
    "AwayTablePos",      
    "TablePosGap",       
]

TEAM_NAME_MAP = {
    "Spurs": "Tottenham Hotspur",
    "Man City": "Man City",
    "Man Utd": "Man United",
    "Nott'm Forest": "Nott'm Forest",
    "Wolves": "Wolves",
}

# Map CSV names to API names
API_NAME_MAP = {
    "Tottenham Hotspur": "Tottenham Hotspur FC",
    "Man City": "Manchester City FC",
    "Man United": "Manchester United FC",
    "Nott'm Forest": "Nottingham Forest FC",
    "Wolves": "Wolverhampton Wanderers FC",
    "Brighton": "Brighton & Hove Albion FC",
    "West Ham": "West Ham United FC",
    "Newcastle": "Newcastle United FC",
    "Aston Villa": "Aston Villa FC",
    "Arsenal": "Arsenal FC",
    "Liverpool": "Liverpool FC",
    "Chelsea": "Chelsea FC",
    "Brentford": "Brentford FC",
    "Everton": "Everton FC",
    "Bournemouth": "AFC Bournemouth",
    "Fulham": "Fulham FC",
    "Sunderland": "Sunderland AFC",
    "Crystal Palace": "Crystal Palace FC",
    "Leeds": "Leeds United FC",
    "Burnley": "Burnley FC",
}


def build_prediction_features(df, home, away):
    """
    Build prediction feature vector using last 5 games of each team.
    Compatible with E0.csv column names.
    """

    # Normalize team names
    home = TEAM_NAME_MAP.get(home, home)
    away = TEAM_NAME_MAP.get(away, away)

    # Last 5 games for home team (all games, not just home)
    home_all = df[(df["HomeTeam"] == home) | (df["AwayTeam"] == home)].tail(5)

    # Last 5 games for away team (all games, not just away)
    away_all = df[(df["HomeTeam"] == away) | (df["AwayTeam"] == away)].tail(5)

    # If not enough history → still allow prediction
    if len(home_all) < 3 or len(away_all) < 3:
        return None

    # Goals scored (home team in any game)
    home_goals_avg = home_all.apply(
        lambda r: r["FullTimeHomeTeamGoals"] if r["HomeTeam"] == home else r["FullTimeAwayTeamGoals"], axis=1
    ).mean()

    away_goals_avg = away_all.apply(
        lambda r: r["FullTimeHomeTeamGoals"] if r["HomeTeam"] == away else r["FullTimeAwayTeamGoals"], axis=1
    ).mean()

    # Goals conceded
    home_conceded_avg = home_all.apply(
        lambda r: r["FullTimeAwayTeamGoals"] if r["HomeTeam"] == home else r["FullTimeHomeTeamGoals"], axis=1
    ).mean()

    away_conceded_avg = away_all.apply(
        lambda r: r["FullTimeAwayTeamGoals"] if r["HomeTeam"] == away else r["FullTimeHomeTeamGoals"], axis=1
    ).mean()

    # Form points (across all games)
    def get_points(row, team):
        if row["HomeTeam"] == team:
            return row["HomeTeamPoints"]
        else:
            return row["AwayTeamPoints"]

    home_form = home_all.apply(lambda r: get_points(r, home), axis=1).sum()
    away_form = away_all.apply(lambda r: get_points(r, away), axis=1).sum()

    # Closeness
    form_closeness = abs(home_form - away_form)
    goals_closeness = abs(home_goals_avg - away_goals_avg)

    # Draw rates
    home_draw_rate = (home_all["FullTimeResult"] == 'D').sum() / len(home_all)
    away_draw_rate = (away_all["FullTimeResult"] == 'D').sum() / len(away_all)

    # Position based on recent form ranking across all teams
    all_teams_home = df.groupby("HomeTeam").tail(5).groupby("HomeTeam")["HomeTeamPoints"].sum()
    all_teams_away = df.groupby("AwayTeam").tail(5).groupby("AwayTeam")["AwayTeamPoints"].sum()

    team_form_dict = {}
    for team in set(list(all_teams_home.index) + list(all_teams_away.index)):
        home_pts = all_teams_home.get(team, 0)
        away_pts = all_teams_away.get(team, 0)
        team_form_dict[team] = home_pts + away_pts

    team_forms = pd.Series(team_form_dict)
    team_positions = team_forms.rank(ascending=False, pct=True)

    home_position = team_positions.get(home, 0.5)
    away_position = team_positions.get(away, 0.5)
    position_gap = abs(home_position - away_position)


    standings = get_current_standings()
    
    # Map to API names
    api_home = API_NAME_MAP.get(home, home + " FC")
    api_away = API_NAME_MAP.get(away, away + " FC")
    
    # Get positions (1-20) and normalize to 0-1
    home_table_pos = standings.get(api_home, 10) / 20.0  # Default mid-table
    away_table_pos = standings.get(api_away, 10) / 20.0
    table_pos_gap = abs(home_table_pos - away_table_pos)

    return pd.DataFrame([{
        "HomeForm5": home_form,
        "AwayForm5": away_form,
        "HomeGoalsAvg": home_goals_avg,
        "AwayGoalsAvg": away_goals_avg,
        "HomeConcededAvg": home_conceded_avg,
        "AwayConcededAvg": away_conceded_avg,
        "FormCloseness": form_closeness,
        "GoalsCloseness": goals_closeness,
        "HomeDrawRate": home_draw_rate,
        "AwayDrawRate": away_draw_rate,
        "HomePosition": home_position,
        "AwayPosition": away_position,
        "PositionGap": position_gap,
        "HomeTablePos": home_table_pos,      
        "AwayTablePos": away_table_pos,      
        "TablePosGap": table_pos_gap,        
    }])[FEATURE_COLUMNS]