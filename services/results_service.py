import requests
from datetime import datetime, timedelta, timezone

API_KEY = "73dfd402f27440d4aff1f6d50185fb3a"
BASE_URL = "https://api.football-data.org/v4/competitions/PL/matches"
HEADERS = {"X-Auth-Token": API_KEY}

OUTCOME_MAP = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win"
}

def fetch_finished_matches():
    # Fetch all finished matches
    url = f"{BASE_URL}?status=FINISHED"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    all_matches = r.json()["matches"]
    
    # Filter to last 14 days in Python (timezone-aware)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=14)
    recent_matches = []
    
    for match in all_matches:
        match_date = datetime.fromisoformat(match['utcDate'].replace('Z', '+00:00'))
        if match_date >= cutoff_date:
            recent_matches.append(match)
    
    return recent_matches


def extract_result(match):
    res = match["score"]["fullTime"]
    if res["home"] > res["away"]:
        return "Home Win"
    elif res["home"] < res["away"]:
        return "Away Win"
    else:
        return "Draw"


def extract_score(match):
    score = match["score"]["fullTime"]
    return score["home"], score["away"]