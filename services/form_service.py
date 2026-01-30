import pandas as pd


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds fast rolling form features:
    HomeForm5 and AwayForm5

    Form points:
    Win = 3
    Draw = 1
    Loss = 0
    """

    print("Computing rolling form features...")

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    team_history = {}
    home_forms = []
    away_forms = []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        # Last 5 matches points
        home_form = sum(team_history.get(home, [])[-5:])
        away_form = sum(team_history.get(away, [])[-5:])

        home_forms.append(home_form)
        away_forms.append(away_form)

        # Update team history after match result
        result = row["FullTimeResult"]

        if result == "H":
            team_history.setdefault(home, []).append(3)
            team_history.setdefault(away, []).append(0)

        elif result == "A":
            team_history.setdefault(home, []).append(0)
            team_history.setdefault(away, []).append(3)

        else:  # Draw
            team_history.setdefault(home, []).append(1)
            team_history.setdefault(away, []).append(1)

    # Add new columns
    df["HomeForm5"] = home_forms
    df["AwayForm5"] = away_forms

    return df
