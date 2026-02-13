import requests
import pandas as pd
from datetime import datetime
import shutil

# API setup
API_KEY = "73dfd402f27440d4aff1f6d50185fb3a"
BASE_URL = "https://api.football-data.org/v4/competitions/PL/matches"
HEADERS = {"X-Auth-Token": API_KEY}

print("Fetching all 2025-26 season matches from API...")

# Fetch ALL matches from current season
url = f"{BASE_URL}?season=2025"
response = requests.get(url, headers=HEADERS, timeout=30)
response.raise_for_status()

matches = response.json()["matches"]
finished_matches = [m for m in matches if m['status'] == 'FINISHED']

print(f"Found {len(finished_matches)} finished matches from 2025-26 season")

# Load existing CSV
df_existing = pd.read_csv("data/premier_league.csv")
print(f"Existing CSV has {len(df_existing)} matches")
print(f"Latest date in CSV: {df_existing['Date'].max()}")

# Convert API matches to CSV format
new_rows = []

for match in finished_matches:
    home_team = match['homeTeam']['name']
    away_team = match['awayTeam']['name']
    
    # Remove FC/AFC suffixes to match CSV naming
    home_team = home_team.replace(" FC", "").replace(" AFC", "")
    away_team = away_team.replace(" FC", "").replace(" AFC", "")
    
    # Extract scores
    home_goals = match['score']['fullTime']['home']
    away_goals = match['score']['fullTime']['away']
    
    # Determine result
    if home_goals > away_goals:
        result = 'H'
    elif away_goals > home_goals:
        result = 'A'
    else:
        result = 'D'
    
    # Parse date
    match_date = datetime.fromisoformat(match['utcDate'].replace('Z', '+00:00'))
    date_str = match_date.strftime('%Y-%m-%d')
    
    # Create row matching CSV structure
    row = {
        'MatchID': f"2025-2026_{home_team}_{away_team}",
        'Season': '2025-2026',
        'MatchWeek': match.get('matchday', ''),
        'Date': date_str,
        'Time': '',
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'FullTimeHomeTeamGoals': home_goals,
        'FullTimeAwayTeamGoals': away_goals,
        'FullTimeResult': result,
        'HalfTimeHomeTeamGoals': '',
        'HalfTimeAwayTeamGoals': '',
        'HalfTimeResult': '',
        'Referee': match.get('referees', [{}])[0].get('name', '') if match.get('referees') else '',
        'HomeTeamShots': '',
        'AwayTeamShots': '',
        'HomeTeamShotsOnTarget': '',
        'AwayTeamShotsOnTarget': '',
        'HomeTeamCorners': '',
        'AwayTeamCorners': '',
        'HomeTeamFouls': '',
        'AwayTeamFouls': '',
        'HomeTeamYellowCards': '',
        'AwayTeamYellowCards': '',
        'HomeTeamRedCards': '',
        'AwayTeamRedCards': '',
        'B365HomeTeam': '',
        'B365Draw': '',
        'B365AwayTeam': '',
        'B365Over2.5Goals': '',
        'B365Under2.5Goals': '',
        'MarketMaxHomeTeam': '',
        'MarketMaxDraw': '',
        'MarketMaxAwayTeam': '',
        'MarketAvgHomeTeam': '',
        'MarketAvgDraw': '',
        'MarketAvgAwayTeam': '',
        'MarketMaxOver2.5Goals': '',
        'MarketMaxUnder2.5Goals': '',
        'MarketAvgOver2.5Goals': '',
        'MarketAvgUnder2.5Goals': '',
        'HomeTeamPoints': 3 if result == 'H' else (1 if result == 'D' else 0),
        'AwayTeamPoints': 3 if result == 'A' else (1 if result == 'D' else 0),
    }
    
    new_rows.append(row)

df_new = pd.DataFrame(new_rows)

print(f"\nNew matches to add: {len(df_new)}")
print(f"Date range: {df_new['Date'].min()} to {df_new['Date'].max()}")

print("\nSample of new matches:")
print(df_new[['Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult', 'MatchWeek']].head(10))

# Combine
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
df_combined['Date'] = pd.to_datetime(df_combined['Date'])
df_combined = df_combined.sort_values('Date')

print(f"\nTotal matches after update: {len(df_combined)}")

# Backup
shutil.copy("data/premier_league.csv", "data/premier_league_backup.csv")
print("Backup saved to: data/premier_league_backup.csv")

# Save
df_combined.to_csv("data/premier_league.csv", index=False)
print("✓ Updated CSV saved!")

print("\n" + "="*60)
print("CSV UPDATE COMPLETE")
print("="*60)
print(f"Old: {len(df_existing)} matches")
print(f"New: {len(df_combined)} matches")
print(f"Added: {len(df_new)} matches from 2025-26 season")
print("="*60)
