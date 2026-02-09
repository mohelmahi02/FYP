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

    total = int(row[0] or 0)  # First column
    correct = int(row[1] or 0)  # Second column

    if total == 0:
        return 0.0, 0, 0

    accuracy = round((correct / total) * 100, 2)
    return accuracy, correct, total


if __name__ == "__main__":
    acc, correct, total = compute_accuracy()
    print(f"Model accuracy: {acc}%")
    print(f"Correct predictions: {correct}/{total}")