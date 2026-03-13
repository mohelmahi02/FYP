import requests
import os

def get_full_standings():
    """Fetch current Premier League standings with full stats from API"""
    
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    
    if not api_key:
        print("Warning: FOOTBALL_DATA_API_KEY not set")
        return []
    
    url = "https://api.football-data.org/v4/competitions/PL/standings"
    headers = {"X-Auth-Token": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            standings = data["standings"][0]["table"]
            
            print(f"✓ Fetched full standings for {len(standings)} teams")
            return standings
        else:
            print(f"Failed to fetch standings: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching standings: {e}")
        return []

def get_current_standings():
    """Fetch current Premier League standings as position map (for predictions)"""
    
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
            
            # Create dictionary: team name -> position 
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