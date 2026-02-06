from services.results_service import fetch_finished_matches, extract_result,extract_score
from services.db_service import get_recent_predictions, update_prediction_result

print("Fetching finished matches from API...")
finished = fetch_finished_matches()

print("Fetching stored predictions...")
predictions = get_recent_predictions(200)

updated = 0

for match in finished:
    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]
    utc_date = match["utcDate"]

    home_goals = match["score"]["fullTime"]["home"]
    away_goals = match["score"]["fullTime"]["away"]

    # Decide actual result
    if home_goals > away_goals:
        actual = "Home Win"
    elif home_goals < away_goals:
        actual = "Away Win"
    else:
        actual = "Draw"

    prediction = None
    for p in predictions:
        if (
            p["home_team"] == home and
            p["away_team"] == away and
            p["utc_date"] == utc_date
        ):
            prediction = p["prediction"]
            break

    if prediction is None:
        continue

    correct = (prediction == actual)

    update_prediction_result(
        home,
        away,
        utc_date,
        actual,
        home_goals,
        away_goals,
        correct
    )

    updated += 1

print(f"Updated {updated} predictions")