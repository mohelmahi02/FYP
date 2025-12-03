#make match outcome predictions using trained model

import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Get API data
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

print(f" Using {'TEST' if USE_TEST_API else 'REAL'} API")
print(f" Total matches: {len(matches)}")


#helper functions

def calculate_team_stats(matches, team_name):
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
    
    return (goals_scored / games_played, goals_conceded / games_played, wins)

def get_match_outcome(home_goals, away_goals):
    if home_goals > away_goals:
        return 2
    elif home_goals == away_goals:
        return 1
    else:
        return 0

#build training data and model

print("Preparing model...")
training_data = []

for i, match in enumerate(matches):
    home_team = match['homeTeam']['name']
    away_team = match['awayTeam']['name']
    home_goals = match['score']['fullTime']['home']
    away_goals = match['score']['fullTime']['away']
    
    if home_goals is None or away_goals is None:
        continue
    
    previous_matches = matches[:i]
    
    if len(previous_matches) < 5:
        continue
    
    home_stats = calculate_team_stats(previous_matches, home_team)
    away_stats = calculate_team_stats(previous_matches, away_team)
    
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

df = pd.DataFrame(training_data)

X = df.drop('outcome', axis=1)
y = df['outcome']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print(" Model trained!")

#make predictions for a new match


print("\n" + "="*50)
print(" MATCH PREDICTIONS")
print("="*50)

# Predict Arsenal vs Chelsea
team1 = "Arsenal FC"
team2 = "Chelsea FC"

team1_stats = calculate_team_stats(matches, team1)
team2_stats = calculate_team_stats(matches, team2)

new_match = pd.DataFrame({
    'home_goals_avg': [team1_stats[0]],
    'home_conceded_avg': [team1_stats[1]],
    'home_wins': [team1_stats[2]],
    'away_goals_avg': [team2_stats[0]],
    'away_conceded_avg': [team2_stats[1]],
    'away_wins': [team2_stats[2]]
})

print(f"\n {team1} (home) vs {team2} (away)")
print(f"\nTeam Stats:")
print(f"  {team1}: {team1_stats[0]:.2f} goals/game, {team1_stats[2]} wins")
print(f"  {team2}: {team2_stats[0]:.2f} goals/game, {team2_stats[2]} wins")

prediction = model.predict(new_match)[0]
probabilities = model.predict_proba(new_match)[0]

outcome_names = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

print(f"\n PREDICTION: {outcome_names[prediction]}")
print(f"\n Confidence:")
print(f"   Home Win: {probabilities[2]*100:5.1f}%")
print(f"   Draw:     {probabilities[1]*100:5.1f}%")
print(f"   Away Win: {probabilities[0]*100:5.1f}%")

print("\n" + "="*50)