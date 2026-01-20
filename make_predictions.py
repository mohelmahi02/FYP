
# Predict upcoming Premier League matches using saved best model (Logistic Regression)

import os
import pickle
import requests
import pandas as pd
from datetime import datetime,timezone

REAL_API_KEY = "73dfd402f27440d4aff1f6d50185fb3a"
BASE_URL = "https://api.football-data.org/v4/competitions/PL/matches"

HEADERS = {"X-Auth-Token": REAL_API_KEY}

MODEL_PATH = os.path.join("models", "logistic_regression.pkl")

outcome_names = {0: "Away Win", 1: "Draw", 2: "Home Win"}


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
            if home_goals > away_goals:
                wins += 1
            games_played += 1

        elif away_team == team_name:
            goals_scored += away_goals
            goals_conceded += home_goals
            if away_goals > home_goals:
                wins += 1
            games_played += 1

    if games_played == 0:
        return 0.0, 0.0, 0

    return (goals_scored / games_played, goals_conceded / games_played, wins)

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
            if team_won:
                points += 3

        games += 1

    return points


def fetch_matches(status):
    url = f"{BASE_URL}?status={status}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["matches"]


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model.py and save the model first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_features(finished_matches, home_team, away_team):
    home_goals_avg, home_conceded_avg, home_wins = calculate_team_stats(finished_matches, home_team)
    away_goals_avg, away_conceded_avg, away_wins = calculate_team_stats(finished_matches, away_team)

    home_goal_diff = home_goals_avg - home_conceded_avg
    away_goal_diff = away_goals_avg - away_conceded_avg

    goal_diff_diff = home_goal_diff - away_goal_diff
    wins_diff = home_wins - away_wins

    home_form_points_5 = calculate_form_points(finished_matches, home_team, last_n=5)
    away_form_points_5 = calculate_form_points(finished_matches, away_team, last_n=5)

    return pd.DataFrame({
        "home_goals_avg": [home_goals_avg],
        "home_conceded_avg": [home_conceded_avg],
        "home_wins": [home_wins],
        "away_goals_avg": [away_goals_avg],
        "away_conceded_avg": [away_conceded_avg],
        "away_wins": [away_wins],

        "home_goal_diff": [home_goal_diff],
        "away_goal_diff": [away_goal_diff],

        "goal_diff_diff": [goal_diff_diff],
        "wins_diff": [wins_diff],

        "home_form_points_5": [home_form_points_5],
        "away_form_points_5": [away_form_points_5],
    })




def predict_upcoming(limit=20):
    print("Loading saved model...")
    model = load_model()
    print("✓ Model loaded:", type(model).__name__)

    print("Fetching finished matches for stats...")
    finished_matches = fetch_matches("FINISHED")
    print(f"✓ Finished matches: {len(finished_matches)}")

    print("Fetching scheduled matches...")
    scheduled_matches = fetch_matches("SCHEDULED")
    print(f"✓ Scheduled matches: {len(scheduled_matches)}")

    predictions = []

    for match in scheduled_matches[:limit]:
        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]
        utc_date = match.get("utcDate")  # ISO string

        X_new = build_features(finished_matches, home_team, away_team)

        pred_class = int(model.predict(X_new)[0])
        probs = model.predict_proba(X_new)[0]

        predictions.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "utc_date": utc_date,
                "prediction": outcome_names[pred_class],
                "home_win_prob": float(probs[2]),
                "draw_prob": float(probs[1]),
                "away_win_prob": float(probs[0]),
                "model_used": "Logistic Regression",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return predictions


if __name__ == "__main__":
    preds = predict_upcoming(limit=10)

    print("\n" + "=" * 60)
    print("UPCOMING MATCH PREDICTIONS (Top 10)")
    print("=" * 60)

    for p in preds:
        print(f"\n{p['home_team']} vs {p['away_team']}")
        print(f"Date (UTC): {p['utc_date']}")
        print(f"Prediction: {p['prediction']}")
        print(
            f"Confidence -> Home: {p['home_win_prob']*100:.1f}% | "
            f"Draw: {p['draw_prob']*100:.1f}% | "
            f"Away: {p['away_win_prob']*100:.1f}%"
        )

    print("\n" + "=" * 60)
