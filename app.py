

from flask import Flask, jsonify, request
from datetime import datetime, timezone
import os
import pickle
import requests
import pandas as pd
from services.model_service import get_model_bundle
from services.db_service import init_db, save_prediction, get_recent_predictions
from services.model_service import get_model_bundle
from services.db_service import init_db, save_prediction, list_predictions
from services.feature_service import build_features



app = Flask(__name__)
init_db()


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

@app.get("/api/history")
def history():
    limit = request.args.get("limit", default=20, type=int)
    limit = max(1, min(limit, 200))
    rows = get_recent_predictions(limit)
    return jsonify({"count": len(rows), "predictions": rows})



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

    bundle = get_model_bundle()
    model = bundle["model"]
    model_name = bundle["best_model_name"]

    finished = fetch_matches("FINISHED")
    X = build_features(finished, home, away)

    pred_class = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]

    result = {
        "home_team": home,
        "away_team": away,
        "prediction": OUTCOME_NAMES[pred_class],
        "home_win_prob": float(probs[2]),
        "draw_prob": float(probs[1]),
        "away_win_prob": float(probs[0]),
        "model_used": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    save_prediction(result)
    return jsonify(result)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
