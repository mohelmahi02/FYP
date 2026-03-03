import psycopg2

# Render EXTERNAL database URL
DATABASE_URL = "postgresql://fyp_user:QBNHDXFxOrXfGHLX6waDOAXsqcrgwe79@dpg-d6jdf77gi27c73d3b580-a.oregon-postgres.render.com/fyp_e26g"

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

# Create table
print("Creating predictions table...")
cur.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    home_team TEXT,
    away_team TEXT,
    utc_date TIMESTAMPTZ,
    prediction TEXT,
    home_win_prob DOUBLE PRECISION,
    draw_prob DOUBLE PRECISION,
    away_win_prob DOUBLE PRECISION,
    model_used TEXT,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    actual_result TEXT,
    correct BOOLEAN,
    home_goals INTEGER,
    away_goals INTEGER,
    gameweek INTEGER
);
""")

print("✓ Table created!")

# Insert GW27 data
print("Inserting GW27 predictions...")
gw27 = [
    ("Man City", "Newcastle", "2026-02-21 12:30:00", "Home Win", 0.671, 0.190, 0.139, "Home Win", 2, 1, True, 27),
    ("Aston Villa", "Leeds", "2026-02-21 15:00:00", "Home Win", 0.598, 0.232, 0.170, "Draw", 1, 1, False, 27),
    ("Brentford", "Brighton", "2026-02-21 15:00:00", "Home Win", 0.399, 0.280, 0.321, "Away Win", 0, 2, False, 27),
    ("Chelsea", "Burnley", "2026-02-21 15:00:00", "Home Win", 0.561, 0.242, 0.198, "Draw", 1, 1, False, 27),
    ("Nott'm Forest", "Liverpool", "2026-02-21 15:00:00", "Away Win", 0.320, 0.269, 0.410, "Away Win", 0, 1, True, 27),
    ("West Ham", "Bournemouth", "2026-02-21 17:30:00", "Away Win", 0.355, 0.285, 0.360, "Draw", 0, 0, False, 27),
    ("Crystal Palace", "Wolves", "2026-02-22 14:00:00", "Away Win", 0.284, 0.263, 0.453, "Home Win", 1, 0, False, 27),
    ("Sunderland", "Fulham", "2026-02-22 14:00:00", "Home Win", 0.530, 0.258, 0.212, "Away Win", 1, 3, False, 27),
    ("Spurs", "Arsenal", "2026-02-22 16:30:00", "Away Win", 0.287, 0.266, 0.446, "Away Win", 1, 4, True, 27),
    ("Everton", "Man Utd", "2026-02-23 20:00:00", "Home Win", 0.444, 0.295, 0.261, "Away Win", 0, 1, False, 27),
]

for p in gw27:
    cur.execute("""
        INSERT INTO predictions 
        (home_team, away_team, utc_date, prediction, home_win_prob, draw_prob, away_win_prob, 
         actual_result, home_goals, away_goals, correct, gameweek, model_used, generated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'Logistic Regression', '2026-02-20 10:00:00')
    """, p)

print("✓ GW27 inserted (3/10 correct = 30%)!")

# Insert GW28 data
print("Inserting GW28 predictions...")
gw28 = [
    ("Wolves", "Aston Villa", "2026-02-27 20:00:00", "Home Win", 0.481, 0.271, 0.248, "Home Win", 2, 0, True, 28),
    ("Bournemouth", "Sunderland", "2026-02-28 12:30:00", "Home Win", 0.602, 0.230, 0.168, "Draw", 1, 1, False, 28),
    ("Brighton", "Nott'm Forest", "2026-02-28 15:00:00", "Draw", 0.379, 0.285, 0.335, "Home Win", 2, 1, False, 28),
    ("Burnley", "Brentford", "2026-02-28 15:00:00", "Away Win", 0.278, 0.256, 0.466, "Away Win", 3, 4, True, 28),
    ("Liverpool", "West Ham", "2026-02-28 15:00:00", "Home Win", 0.487, 0.269, 0.243, "Home Win", 5, 2, True, 28),
    ("Newcastle", "Everton", "2026-02-28 15:00:00", "Home Win", 0.542, 0.241, 0.217, "Away Win", 2, 3, False, 28),
    ("Leeds", "Man City", "2026-02-28 17:30:00", "Draw", 0.368, 0.277, 0.355, "Away Win", 0, 1, False, 28),
    ("Fulham", "Spurs", "2026-03-01 14:00:00", "Home Win", 0.620, 0.219, 0.161, "Home Win", 2, 1, True, 28),
    ("Man Utd", "Crystal Palace", "2026-03-01 14:00:00", "Draw", 0.360, 0.284, 0.356, "Home Win", 2, 1, False, 28),
    ("Arsenal", "Chelsea", "2026-03-01 16:30:00", "Away Win", 0.300, 0.264, 0.436, "Home Win", 2, 1, False, 28),
]

for p in gw28:
    cur.execute("""
        INSERT INTO predictions 
        (home_team, away_team, utc_date, prediction, home_win_prob, draw_prob, away_win_prob, 
         actual_result, home_goals, away_goals, correct, gameweek, model_used, generated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'Logistic Regression', '2026-02-27 10:00:00')
    """, p)

print("✓ GW28 inserted (4/10 correct = 40%)!")

conn.commit()
cur.close()
conn.close()

print("\n Database initialized!")
print("Refresh https://fyp-frontend-wgcl.onrender.com to see predictions!")
