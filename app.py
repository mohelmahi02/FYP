

from flask import Flask, jsonify, request
from datetime import datetime, timezone
import os
import pickle
import requests
import pandas as pd

app = Flask(__name__)

API_KEY = "73dfd402f27440d4aff1f6d50185fb3a"
BASE_URL = "https://api.football-data.org/v4/competitions/PL/matches"
HEADERS = {"X-Auth-Token": API_KEY}

MODEL_PATH = os.path.join("models", "logistic_regression.pkl")
RESULTS_PATH = os.path.join("models", "model_results.pkl")

OUTCOME_NAMES = {0: "Away Win", 1: "Draw", 2: "Home Win"}


def fetch_matches(status: str):
    url = f"{BASE_URL}?status={status}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["matches"]


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}. Run train_model.py to generate it."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH, "rb") as f:
        return pickle.load(f)


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


def build_features(finished_matches, home_team, away_team):
    home_stats = calculate_team_stats(finished_matches, home_team)
    away_stats = calculate_team_stats(finished_matches, away_team)

    return pd.DataFrame(
        {
            "home_goals_avg": [home_stats[0]],
            "home_conceded_avg": [home_stats[1]],
            "home_wins": [home_stats[2]],
            "away_goals_avg": [away_stats[0]],
            "away_conceded_avg": [away_stats[1]],
            "away_wins": [away_stats[2]],
        }
    )


def predict_match(model, finished_matches, home_team, away_team):
    X_new = build_features(finished_matches, home_team, away_team)
    pred_class = int(model.predict(X_new)[0])
    probs = model.predict_proba(X_new)[0]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "prediction": OUTCOME_NAMES[pred_class],
        "home_win_prob": float(probs[2]),
        "draw_prob": float(probs[1]),
        "away_win_prob": float(probs[0]),
        "model_used": "Logistic Regression",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


@app.get("/api/models")
def models():
    results = load_results()
    if results is None:
        return jsonify({"error": "model_results.pkl not found. Run train_model.py first."}), 404
    return jsonify(results)


@app.get("/api/upcoming")
def upcoming():
    limit = request.args.get("limit", default=10, type=int)
    limit = max(1, min(limit, 50))

    model = load_model()
    finished = fetch_matches("FINISHED")
    scheduled = fetch_matches("SCHEDULED")

    predictions = []
    for match in scheduled[:limit]:
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        utc_date = match.get("utcDate")

        p = predict_match(model, finished, home, away)
        p["utc_date"] = utc_date
        predictions.append(p)

    return jsonify({"count": len(predictions), "predictions": predictions})


@app.get("/api/predict")
def predict():
    home = request.args.get("home")
    away = request.args.get("away")
    if not home or not away:
        return jsonify({"error": "Provide query params: home and away"}), 400

    model = load_model()
    finished = fetch_matches("FINISHED")
    p = predict_match(model, finished, home, away)
    return jsonify(p)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
