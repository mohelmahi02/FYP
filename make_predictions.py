import pickle
import pandas as pd

from services.prediction_feature_service import build_prediction_features
from services.form_service import add_form_features
from services.fixtures_service import fetch_scheduled_matches, get_next_matchweek

MODEL_PATH = "models/logistic_regression.pkl"

OUTCOME_NAMES = {
    0: "Away Win",
    1: "Draw",
    2: "Home Win"
}

print("Loading Kaggle dataset...")
df_data = pd.read_csv("data/premier_league.csv")

print("Adding form features...")
df_data = add_form_features(df_data)

print("Loading trained model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("\nFetching upcoming fixtures...")
all_fixtures = fetch_scheduled_matches()

print("Selecting next matchweek only...")
fixtures = get_next_matchweek(all_fixtures)

print("\nUPCOMING MATCHWEEK PREDICTIONS")
print("=" * 60)

for match in fixtures:
    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]
    date = match["utcDate"]

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
