import os
import psycopg2
from psycopg2.extras import RealDictCursor
from services.db_service import init_db
init_db()


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "fyp_db",
    "user": "fyp_user",
    "password": "fyp_password",
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    """Create tables if they do not exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            prediction TEXT NOT NULL,
            home_win_prob FLOAT,
            draw_prob FLOAT,
            away_win_prob FLOAT,
            model_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cur.close()
    conn.close()


def save_prediction(data: dict):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO predictions (
            home_team, away_team, prediction,
            home_win_prob, draw_prob, away_win_prob,
            model_used
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        data["home_team"],
        data["away_team"],
        data["prediction"],
        data.get("home_win_prob"),
        data.get("draw_prob"),
        data.get("away_win_prob"),
        data.get("model_used"),
    ))

    conn.commit()
    cur.close()
    conn.close()


def get_prediction_history(limit=50):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT *
        FROM predictions
        ORDER BY created_at DESC
        LIMIT %s
    """, (limit,))

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows
