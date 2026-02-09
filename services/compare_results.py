from datetime import datetime, timezone

from services.db_service import get_pending_predictions, update_prediction_result
from services.results_service import fetch_finished_matches  


TEAM_NAME_MAP = {
    "Manchester United FC": "Man Utd",
    "Manchester City FC": "Man City",
    "Tottenham Hotspur FC": "Spurs",
    "Nottingham Forest FC": "Nott'm Forest",
    "AFC Bournemouth": "Bournemouth",
    "Newcastle United FC": "Newcastle",
    "Brighton & Hove Albion FC": "Brighton",
    "Wolverhampton Wanderers FC": "Wolves",
    "West Ham United FC": "West Ham",
    "Leeds United FC": "Leeds",
    "Fulham FC": "Fulham",
    "Liverpool FC": "Liverpool",
    "Everton FC": "Everton",
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea",
    "Brentford FC": "Brentford",
    "Crystal Palace FC": "Crystal Palace",
    "Sunderland AFC": "Sunderland",
    "Aston Villa FC": "Aston Villa",
    "Burnley FC": "Burnley",
}


def normalise_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def to_iso_utc(dt_str: str) -> str:
    """
    Ensure API utcDate matches DB utc_date format.
    DB utc_date in your table looked like: '2026-02-01T14:00:00Z' (text).
    """
    # If already ends with Z, keep it
    if isinstance(dt_str, str) and dt_str.endswith("Z"):
        return dt_str
    return dt_str


def extract_score(match):
    """
    Works for football-data style: match['score']['fullTime']['home'] / ['away']
    """
    ft = match.get("score", {}).get("fullTime", {})
    home_goals = ft.get("home")
    away_goals = ft.get("away")
    return home_goals, away_goals


def extract_result(match):
    home_goals, away_goals = extract_score(match)
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "Home Win"
    if away_goals > home_goals:
        return "Away Win"
    return "Draw"


def find_matching_match(finished_matches, home_team, away_team, utc_date):
    """
    Matches DB row to API row by home/away team + utc date.
    """
    # Convert DB datetime to string with T and Z to match API format
    if isinstance(utc_date, str):
        
        db_date_str = utc_date.replace(' ', 'T') + 'Z'
    else:
        
        db_date_str = utc_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    for m in finished_matches:
        api_home = normalise_team(m.get("homeTeam", {}).get("name", ""))
        api_away = normalise_team(m.get("awayTeam", {}).get("name", ""))
        api_date = m.get("utcDate")

        if api_home == home_team and api_away == away_team and api_date == db_date_str:
            return m

    return None


def main():
    print("Fetching finished matches from API...")
    finished_matches = fetch_finished_matches()

    print("Fetching stored predictions...")
    predictions = get_pending_predictions()

    updated = 0
    total_considered = 0

    for p in predictions:
       
        
        home_team, away_team, utc_date, prediction = p

        match = find_matching_match(finished_matches, home_team, away_team, utc_date)
        if not match:
            continue

        actual_result = extract_result(match)
        if actual_result is None:
            continue

        home_goals, away_goals = extract_score(match)
        correct = (prediction == actual_result)

        update_prediction_result(
            home_team=home_team,
            away_team=away_team,
            utc_date=utc_date,
            actual_result=actual_result,
            home_goals=home_goals,
            away_goals=away_goals,
            correct=correct
        )

        updated += 1
        total_considered += 1

    print(f"Updated {updated} predictions")


    if total_considered > 0:
        
        pass


if __name__ == "__main__":
    main()
