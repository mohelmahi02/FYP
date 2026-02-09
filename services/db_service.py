import psycopg2
import os

def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("POSTGRES_DB", "fyp"),
        user=os.getenv("POSTGRES_USER", "fyp_user"),
        password=os.getenv("POSTGRES_PASSWORD", "fyp_pass"),
        port=5432
    )



# SAVE PREDICTION

def save_prediction(
    home_team,
    away_team,
    utc_date,
    prediction
):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO predictions (
            home_team,
            away_team,
            utc_date,
            prediction
        )
        VALUES (%s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (home_team, away_team, utc_date, prediction)
    )

    conn.commit()
    cur.close()
    conn.close()



# GET PENDING PREDICTIONS

def get_pending_predictions():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            home_team,
            away_team,
            utc_date,
            prediction
        FROM predictions
        WHERE actual_result IS NULL
        """
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return rows



# UPDATE RESULTS

def update_prediction_result(
    home_team,
    away_team,
    utc_date,
    actual_result,
    home_goals,
    away_goals,
    correct
):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE predictions
        SET
            actual_result = %s,
            home_goals = %s,
            away_goals = %s,
            correct = %s
        WHERE
            home_team = %s
            AND away_team = %s
            AND utc_date = %s
        """,
        (
            actual_result,
            home_goals,
            away_goals,
            correct,
            home_team,
            away_team,
            utc_date
        )
    )

    conn.commit()
    cur.close()
    conn.close()
