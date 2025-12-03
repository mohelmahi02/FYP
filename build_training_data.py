
#Build trainig data for ML model

import requests
import pandas as pd

# Get API data
USE_TEST_API = True
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

# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_team_stats(matches, team_name):
    """Calculate team statistics"""
    goals_scored = 0
    goals_conceded = 0
    wins = 0
    games_played = 0
    
    for match in matches:
        home_team = match['homeTeam']['name']
        away_team = match['awayTeam']['name']
        home_goals = match['score']['fullTime']['home']
        away_goals = match['score']['fullTime']['away']
        
        if home_goals is None or away_goals is None:
            continue
            
        if home_team == team_name:
            goals_scored += home_goals
            goals_conceded += away_goals
            if home_goals > away_goals:
                wins += 1
            games_played += 1
            
        elif away_team == team_name:
            goals_scored += away_goals
            goals_conceded += home_goals
            if away_goals > home_goals:
                wins += 1
            games_played += 1
    
    if games_played == 0:
        return 0, 0, 0
    
    return (
        goals_scored / games_played,
        goals_conceded / games_played,
        wins
    )

def get_match_outcome(home_goals, away_goals):
    """Convert score to outcome: 2=Win, 1=Draw, 0=Loss"""
    if home_goals > away_goals:
        return 2  # Win
    elif home_goals == away_goals:
        return 1  # Draw
    else:
        return 0  # Loss

# ============================================
# BUILD TRAINING DATA
# ============================================

print("Building training dataset...")
training_data = []

for i, match in enumerate(matches):
    home_team = match['homeTeam']['name']
    away_team = match['awayTeam']['name']
    home_goals = match['score']['fullTime']['home']
    away_goals = match['score']['fullTime']['away']
    
    if home_goals is None or away_goals is None:
        continue
    
    # Only use previous matches to avoid data leakage
    previous_matches = matches[:i]
    
    # Need at least 5 previous matches for stats
    if len(previous_matches) < 5:
        continue
    
    # Calculate stats for both teams
    home_stats = calculate_team_stats(previous_matches, home_team)
    away_stats = calculate_team_stats(previous_matches, away_team)
    
    # Create feature row
    features = {
        'home_goals_avg': home_stats[0],
        'home_conceded_avg': home_stats[1],
        'home_wins': home_stats[2],
        'away_goals_avg': away_stats[0],
        'away_conceded_avg': away_stats[1],
        'away_wins': away_stats[2],
        'outcome': get_match_outcome(home_goals, away_goals)
    }
    
    training_data.append(features)

# Convert to DataFrame
df = pd.DataFrame(training_data)

print(f" Created {len(df)} training samples")
print("\nFirst 5 samples:")
print(df.head())
print("\nFeatures created:")
print(df.columns.tolist())