import requests
from datetime import datetime, timezone

from services.feature_service import build_features
from services.model_service import get_model_bundle

REAL_API_KEY = "73dfd402f27440d4aff1f6d50185fb3a"
BASE_URL = "https://api.football-data.org/v4/competitions/PL/matches"
HEADERS = {"X-Auth-Token": REAL_API_KEY}

OUTCOME_NAMES = {0: "Away Win", 1: "Draw", 2: "Home Win"}


def fetch_matches(status: str):
    url = f"{BASE_URL}?status={status}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["matches"]

def predict_upcoming(limit=20):
    print("Loading saved model...")
    bundle = get_model_bundle()
    model = bundle["model"]
    model_name = bundle["best_model_name"]
    print("✓ Model loaded:", type(model).__name__, f"({model_name})")

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
        utc_date = match.get("utcDate")

        # uses services.feature_service.build_features
        X_new = build_features(finished_matches, home_team, away_team)

        pred_class = int(model.predict(X_new)[0])
        probs = model.predict_proba(X_new)[0]

        predictions.append({
            "home_team": home_team,
            "away_team": away_team,
            "utc_date": utc_date,
            "prediction": OUTCOME_NAMES[pred_class],
            "home_win_prob": float(probs[2]),
            "draw_prob": float(probs[1]),
            "away_win_prob": float(probs[0]),
            "model_used": model_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

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

