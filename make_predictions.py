import pandas as pd
import pickle
from datetime import datetime, timezone

from services.db_service import init_db, save_prediction
from services.form_service import add_form_features
from services.fixtures_service import load_fixtures, get_next_matchweek
from services.prediction_feature_service import build_prediction_features


MODEL_PATH = "models/logistic_regression.pkl"
DATA_FILE = "data/E0.csv"
FIXTURES_FILE = "data/fixtures_gw24_38.csv"

OUTCOME_NAMES = {
    0: "Away Win",
    1: "Draw",
    2: "Home Win"
}


print("\nInitializing PostgreSQL...")
init_db()

print("\nLoading match history dataset...")
df_data = pd.read_csv(DATA_FILE)

print("Adding form features...")
df_data = add_form_features(df_data)

print("\nLoading trained model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("\nLoading fixtures CSV...")
fixtures_df = load_fixtures(FIXTURES_FILE)

print("\nSelecting next matchweek only...")
next_week = get_next_matchweek(fixtures_df)

print("\nUPCOMING MATCHWEEK PREDICTIONS")
print("=" * 60)

saved = 0

for _, match in next_week.iterrows():
    home = match["HomeTeam"]
    away = match["AwayTeam"]
    date = match["Date"]

    X = build_prediction_features(df_data, home, away)

    if X is None:
        print(f"Skipping {home} vs {away} (not enough history)")
        continue

    pred_class = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]

    result = {
        "home_team": home,
        "away_team": away,
        "utc_date": str(date),
        "prediction": OUTCOME_NAMES[pred_class],
        "home_win_prob": float(probs[2]),
        "draw_prob": float(probs[1]),
        "away_win_prob": float(probs[0]),
        "model_used": "Logistic Regression",
        "generated_at": datetime.now(timezone.utc)
    }

    print(f"\n{home} vs {away}")
    print(f"Date: {date}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence → Home: {result['home_win_prob']*100:.1f}% | "
          f"Draw: {result['draw_prob']*100:.1f}% | "
          f"Away: {result['away_win_prob']*100:.1f}%")

    save_prediction(result)
    saved += 1


print("\n" + "=" * 60)
print(f"Saved {saved} matchweek predictions into PostgreSQL ")
print("=" * 60)
