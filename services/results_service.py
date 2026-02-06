import requests
from datetime import datetime

API_KEY = "73dfd402f27440d4aff1f6d50185fb3a"
BASE_URL = "https://api.football-data.org/v4/competitions/PL/matches"
HEADERS = {"X-Auth-Token": API_KEY}

OUTCOME_MAP = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win"
}

def fetch_finished_matches():
    url = f"{BASE_URL}?status=FINISHED"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["matches"]


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