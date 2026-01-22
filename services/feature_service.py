import pandas as pd


def calculate_team_stats(matches, team_name):
    goals_scored = 0
    goals_conceded = 0
    wins = 0
    games_played = 0

    for match in matches:
        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]
        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]

        if home_goals is None or away_goals is None:
            continue

        if home_team == team_name:
            goals_scored += home_goals
            goals_conceded += away_goals
            wins += 1 if home_goals > away_goals else 0
            games_played += 1

        elif away_team == team_name:
            goals_scored += away_goals
            goals_conceded += home_goals
            wins += 1 if away_goals > home_goals else 0
            games_played += 1

    if games_played == 0:
        return 0.0, 0.0, 0

    return goals_scored / games_played, goals_conceded / games_played, wins


def calculate_form_points(matches, team_name, last_n=5):
    points = 0
    games = 0

    for match in reversed(matches):
        if games >= last_n:
            break

        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]
        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]

        if home_goals is None or away_goals is None:
            continue

        if team_name != home_team and team_name != away_team:
            continue

        if home_goals == away_goals:
            points += 1
        else:
            team_won = (
                (team_name == home_team and home_goals > away_goals)
                or (team_name == away_team and away_goals > home_goals)
            )
            points += 3 if team_won else 0

        games += 1

    return points


def build_features(finished_matches, home_team, away_team):
    home_goals_avg, home_conceded_avg, home_wins = calculate_team_stats(
        finished_matches, home_team
    )
    away_goals_avg, away_conceded_avg, away_wins = calculate_team_stats(
        finished_matches, away_team
    )

    home_goal_diff = home_goals_avg - home_conceded_avg
    away_goal_diff = away_goals_avg - away_conceded_avg

    home_form_points_5 = calculate_form_points(
        finished_matches, home_team, last_n=5
    )
    away_form_points_5 = calculate_form_points(
        finished_matches, away_team, last_n=5
    )

    return pd.DataFrame(
        {
            "home_goals_avg": [home_goals_avg],
            "home_conceded_avg": [home_conceded_avg],
            "home_wins": [home_wins],
            "away_goals_avg": [away_goals_avg],
            "away_conceded_avg": [away_conceded_avg],
            "away_wins": [away_wins],

            # Engineered features
            "home_goal_diff": [home_goal_diff],
            "away_goal_diff": [away_goal_diff],
            "goal_diff_diff": [home_goal_diff - away_goal_diff],
            "wins_diff": [home_wins - away_wins],

            # Recent form
            "home_form_points_5": [home_form_points_5],
            "away_form_points_5": [away_form_points_5],
        }
    )
