#calculate stats

import requests

# Get data from API
USE_TEST_API = False
TEST_API_URL = "https://jsonblob.com/api/jsonBlob/019ae639-062a-7bc8-bd2f-65d55859bb27"
REAL_API_KEY = "73dfd402f27440d4aff1f6d50185fb3a"
REAL_API_URL = "https://api.football-data.org/v4/competitions/PL/matches?status=FINISHED"

if USE_TEST_API:
    response = requests.get(TEST_API_URL)
else:
    headers = {"X-Auth-Token": REAL_API_KEY}
    response = requests.get(REAL_API_URL, headers=headers)

data = response.json()
matches = data['matches']

print(f"Got {len(matches)} matches from API")

#calculate team stats

def calculate_team_stats(matches, team_name):
    """
    Calculate statistics for a team from their match history
    Returns: (goals_per_game, conceded_per_game, total_wins)
    """
    goals_scored = 0
    goals_conceded = 0
    wins = 0
    games_played = 0
    
    for match in matches:
        home_team = match['homeTeam']['name']
        away_team = match['awayTeam']['name']
        home_goals = match['score']['fullTime']['home']
        away_goals = match['score']['fullTime']['away']
        
        # Skip matches with no score
        if home_goals is None or away_goals is None:
            continue
            
        # Check if this team played at home
        if home_team == team_name:
            goals_scored += home_goals
            goals_conceded += away_goals
            if home_goals > away_goals:
                wins += 1
            games_played += 1
            
        # Check if this team played away
        elif away_team == team_name:
            goals_scored += away_goals
            goals_conceded += home_goals
            if away_goals > home_goals:
                wins += 1
            games_played += 1
    
    # Avoid division by zero
    if games_played == 0:
        return 0, 0, 0
    
    # Return averages per game
    return (
        goals_scored / games_played,
        goals_conceded / games_played,
        wins
    )

#test function

print("\nTesting team stats calculation...")

arsenal_stats = calculate_team_stats(matches, "Arsenal FC")
print(f"\nArsenal FC:")
print(f"  Goals per game: {arsenal_stats[0]:.2f}")
print(f"  Conceded per game: {arsenal_stats[1]:.2f}")
print(f"  Total wins: {arsenal_stats[2]}")

chelsea_stats = calculate_team_stats(matches, "Chelsea FC")
print(f"\nChelsea FC:")
print(f"  Goals per game: {chelsea_stats[0]:.2f}")
print(f"  Conceded per game: {chelsea_stats[1]:.2f}")
print(f"  Total wins: {chelsea_stats[2]}")