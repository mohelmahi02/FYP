from services.db_service import get_conn

def compute_accuracy():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS correct
        FROM predictions
        WHERE actual_result IS NOT NULL;
    """)

    row = cur.fetchone()
    conn.close()

    total = int(row["total"] or 0)
    correct = int(row["correct"] or 0)

    if total == 0:
        return 0.0, 0, 0

    accuracy = round((correct / total) * 100, 2)
    return accuracy, correct, total
