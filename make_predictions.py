import os
import pickle
import requests
import pandas as pd
from datetime import datetime, timezone

from services.prediction_feature_service import build_prediction_features
from services.form_service import add_form_features





API_KEY = "73dfd402f27440d4aff1f6d50185fb3a"
BASE_URL = "https://api.football-data.org/v4/competitions/PL/matches"
HEADERS = {"X-Auth-Token": API_KEY}

MODEL_PATH = "models/logistic_regression.pkl"

OUTCOME_NAMES = {0: "Away Win", 1: "Draw", 2: "Home Win"}




print("Loading Kaggle dataset...")
df_data = pd.read_csv("data/premier_league.csv")

print("Adding form features...")
df_data = add_form_features(df_data)


print("Loading trained model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def fetch_upcoming():
    url = f"{BASE_URL}?status=SCHEDULED"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()["matches"]


print("\nFetching upcoming fixtures...")
fixtures = fetch_upcoming()

print("\nUPCOMING MATCH PREDICTIONS")
print("=" * 50)

for match in fixtures[:10]:
    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]
    date = match["utcDate"]


    # Build feature row from Kaggle history
    X = build_prediction_features(df_data, home, away)

    if X is None:
        print(f"Skipping {home} vs {away} (not enough history)")
        continue

    pred_class = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]

    

    print(f"\n{home} vs {away}")
    print("Date:", date)
    print("Prediction:", OUTCOME_NAMES[pred_class])
    print(
        f"Confidence → Home: {probs[2]*100:.1f}% | "
        f"Draw: {probs[1]*100:.1f}% | "
        f"Away: {probs[0]*100:.1f}%"

        
    )
