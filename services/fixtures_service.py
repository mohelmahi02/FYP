import requests

API_KEY = "73dfd402f27440d4aff1f6d50185fb3a"
BASE_URL = "https://api.football-data.org/v4/competitions/PL/matches"
HEADERS = {"X-Auth-Token": API_KEY}


def fetch_scheduled_matches():
    """Fetch all upcoming Premier League fixtures"""
    url = f"{BASE_URL}?status=SCHEDULED"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()["matches"]


def get_next_matchweek(fixtures):
    """
    Return all fixtures from the next matchday (full matchweek)
    """

    if not fixtures:
        return []

    fixtures = sorted(fixtures, key=lambda x: x["utcDate"])

    # Matchday number of the first upcoming fixture
    next_matchday = fixtures[0]["matchday"]

    # Return all fixtures with that matchday number
    matchweek = [m for m in fixtures if m["matchday"] == next_matchday]

    return matchweek
