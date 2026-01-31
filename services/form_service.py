import pandas as pd


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds HomeTeamPoints and AwayTeamPoints based on match result.

    Works for BOTH:
    - Kaggle dataset (FullTimeResult)
    - Football-Data E0.csv dataset (FTR)
    """

    # Detect correct result column
    if "FullTimeResult" in df.columns:
        result_col = "FullTimeResult"
    elif "FTR" in df.columns:
        result_col = "FTR"
    else:
        raise ValueError("No result column found (FullTimeResult or FTR)")

    home_points = []
    away_points = []

    for _, row in df.iterrows():
        result = row[result_col]

        if result == "H":
            home_points.append(3)
            away_points.append(0)
        elif result == "A":
            home_points.append(0)
            away_points.append(3)
        else:
            home_points.append(1)
            away_points.append(1)

    df["HomeTeamPoints"] = home_points
    df["AwayTeamPoints"] = away_points

    return df
