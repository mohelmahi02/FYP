import os
import psycopg2
from psycopg2.extras import RealDictCursor


# Default local/dev connection 
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "fyp")
DB_USER = os.getenv("DB_USER", "fyp_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "fyp_pass")


def get_conn():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        cursor_factory=RealDictCursor,
    )


def init_db():
    """Create tables if they don't exist."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        home_team TEXT NOT NULL,
                        away_team TEXT NOT NULL,
                        utc_date TEXT,
                        prediction TEXT NOT NULL,
                        home_win_prob DOUBLE PRECISION NOT NULL,
                        draw_prob DOUBLE PRECISION NOT NULL,
                        away_win_prob DOUBLE PRECISION NOT NULL,
                        model_used TEXT NOT NULL,
                        generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
    finally:
        conn.close()

def save_prediction(pred):
    """Insert one prediction row into the database"""

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions
                    (home_team, away_team, utc_date, prediction,
                     home_win_prob, draw_prob, away_win_prob,
                     model_used, generated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        pred["home_team"],
                        pred["away_team"],
                        pred["utc_date"],
                        pred["prediction"],
                        pred["home_win_prob"],
                        pred["draw_prob"],
                        pred["away_win_prob"],
                        pred["model_used"],
                        pred["generated_at"],
                    ),
                )
    finally:
        conn.close()


def list_predictions(limit=50):
    """Return last saved predictions"""

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM predictions
                ORDER BY generated_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()
    finally:
        conn.close()





def get_recent_predictions(limit: int = 20):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM predictions
                ORDER BY generated_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            return cur.fetchall()
    finally:
        conn.close()
