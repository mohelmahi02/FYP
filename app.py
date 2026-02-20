from flask import Flask, jsonify, request
from datetime import datetime, timezone
import os
import pickle
import requests
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS 
from datetime import datetime, timezone
from services.fixtures_service import load_fixtures, get_next_matchweek
from services.model_service import get_model_bundle
from services.db_service import (
    save_prediction,
    list_predictions,
    get_conn,
    get_recent_predictions,
)
from services.prediction_feature_service import build_prediction_features


# Flask App Setup

app = Flask(__name__)
CORS(app)
# init_db()


# Constants

API_KEY = "73dfd402f27440d4aff1f6d50185fb3a"
BASE_URL = "https://api.football-data.org/v4/competitions/PL/matches"
HEADERS = {"X-Auth-Token": API_KEY}

OUTCOME_NAMES = {0: "Away Win", 1: "Draw", 2: "Home Win"}


# Load Kaggle Dataset

DATA_PATH = "data/premier_league.csv"

print("Loading Kaggle dataset...")
df_data = pd.read_csv(DATA_PATH)
df_data["Date"] = pd.to_datetime(df_data["Date"])
df_data = df_data.sort_values("Date")
print("✓ Dataset loaded:", len(df_data), "matches")



def fetch_matches(status: str):
    url = f"{BASE_URL}?status={status}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["matches"]




@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat()
    })


@app.get("/api/models")
def models():
    bundle = get_model_bundle()
    return jsonify(bundle["results"])


@app.get("/api/history")
def history():
    limit = request.args.get("limit", default=20, type=int)
    rows = get_recent_predictions(limit)
    return jsonify({"count": len(rows), "predictions": rows})

@app.get("/api/accuracy")
def accuracy():
    from services.evaluation_service import compute_accuracy
    acc, correct, total = compute_accuracy()
    return jsonify({
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

@app.get("/api/predictions")
def predictions():
    limit = request.args.get("limit", default=50, type=int)
    
    conn = get_conn()
    try:
        from psycopg2.extras import RealDictCursor
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM predictions
                WHERE actual_result IS NULL
                ORDER BY utc_date ASC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return jsonify({"count": len(rows), "predictions": rows})
    finally:
        conn.close()

@app.get("/api/predict")
def predict():
    home = (request.args.get("home") or "").strip()
    away = (request.args.get("away") or "").strip()

    if not home or not away:
        return jsonify({"error": "Query parameters 'home' and 'away' are required"}), 400

    if home.lower() == away.lower():
        return jsonify({"error": "Home and away teams must be different"}), 400

    try:
        bundle = get_model_bundle()
        model = bundle["model"]
        model_name = bundle["best_model_name"]

        # Build Kaggle feature row
        X = build_prediction_features(df_data, home, away)

        if X is None:
            return jsonify({"error": "Not enough match history for these teams"}), 400

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

        # Save to Postgres
        save_prediction(result)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.get("/api/matchweek")
def predict_matchweek():
    bundle = get_model_bundle()
    model = bundle["model"]
    model_name = bundle["best_model_name"]

    # Load fixtures
    fixtures_df = load_fixtures("data/fixtures_gw24_38.csv")
    week = get_next_matchweek(fixtures_df, df_data)
    
    if week is None:
        return jsonify({"error": "No upcoming matchweeks found"}), 404

    # Get fixtures for this week
    fixtures = fixtures_df[fixtures_df["Gameweek"] == week]
    
    predictions = []

    for _, match in fixtures.iterrows():
        home = match["HomeTeam"]
        away = match["AwayTeam"]
        date = match["Date"]

        X = build_prediction_features(df_data, home, away)

        if X is None:
            continue

        pred_class = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]

        result = {
            "matchweek": int(week),
            "home_team": home,
            "away_team": away,
            "date": date,
            "prediction": OUTCOME_NAMES[pred_class],
            "home_win_prob": float(probs[2]),
            "draw_prob": float(probs[1]),
            "away_win_prob": float(probs[0]),
            "model_used": model_name,
        }

        save_prediction(result)
        predictions.append(result)

    return jsonify({
        "matchweek": int(week),
        "count": len(predictions),
        "predictions": predictions
    })
#Run flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
