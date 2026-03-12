import requests
import os

def get_current_standings():
    """Fetch current Premier League standings from API"""
    
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    
    if not api_key:
        print("Warning: FOOTBALL_DATA_API_KEY not set, using fallback positions")
        return {}
    
    url = "https://api.football-data.org/v4/competitions/PL/standings"
    headers = {"X-Auth-Token": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            standings = data["standings"][0]["table"]
            
            # Create dictionary: team name -> position (1-20)
            position_map = {}
            for entry in standings:
                team_name = entry["team"]["name"]
                position = entry["position"]
                position_map[team_name] = position
            
            print(f"✓ Fetched standings for {len(position_map)} teams")
            return position_map
        else:
            print(f"Failed to fetch standings: {response.status_code}")
            return {}
    except Exception as e:
        print(f"Error fetching standings: {e}")
        return {}

# Team name mapping for API vs CSV differences
STANDINGS_NAME_MAP = {
    "Tottenham Hotspur": "Tottenham Hotspur FC",
    "Man City": "Manchester City FC",
    "Man United": "Manchester United FC",
    "Nott'm Forest": "Nottingham Forest FC",
    "Wolves": "Wolverhampton Wanderers FC",
    "Brighton": "Brighton & Hove Albion FC",
    "West Ham": "West Ham United FC",
    "Newcastle": "Newcastle United FC",
    "Leicester": "Leicester City FC",
}

if __name__ == "__main__":
    standings = get_current_standings()
    if standings:
        for team, pos in sorted(standings.items(), key=lambda x: x[1]):
            print(f"{pos:2d}. {team}")
