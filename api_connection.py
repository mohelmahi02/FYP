#connection to API
import requests

# Configuration
USE_TEST_API = True

# API URLs
TEST_API_URL = "https://jsonblob.com/api/jsonBlob/019ae639-062a-7bc8-bd2f-65d55859bb27"
REAL_API_KEY = "73dfd402f27440d4aff1f6d50185fb3a"
REAL_API_URL = "https://api.football-data.org/v4/competitions/PL/matches?status=FINISHED"

# Connect to API
print(f"🔌 Connecting to {'TEST' if USE_TEST_API else 'REAL'} API...")

if USE_TEST_API:
    response = requests.get(TEST_API_URL)
else:
    headers = {"X-Auth-Token": REAL_API_KEY}
    response = requests.get(REAL_API_URL, headers=headers)

# Check response
if response.status_code == 200:
    data = response.json()
    print(f" Connected successfully!")
    print(f" Total matches retrieved: {len(data['matches'])}")
    
    # Show first match as example
    first_match = data['matches'][0]
    print(f"\nExample match:")
    print(f"{first_match['homeTeam']['name']} vs {first_match['awayTeam']['name']}")
    print(f"Score: {first_match['score']['fullTime']['home']}-{first_match['score']['fullTime']['away']}")
else:
    print(f" Error: {response.status_code}")