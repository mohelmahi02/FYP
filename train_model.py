#train random forest model

import os 
import pickle
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# real football-data.org API
API_KEY = "73dfd402f27440d4aff1f6d50185fb3a"
API_URL = "https://api.football-data.org/v4/competitions/PL/matches"

headers = {"X-Auth-Token": API_KEY}
response = requests.get(API_URL, headers=headers)
data = response.json()

print(f"Fetched {len(data['matches'])} matches from API")

matches = data['matches']
#functions
def calculate_team_stats(matches, team_name):
    goals_scored = 0
    goals_conceded = 0
    wins = 0
    games_played = 0
    
    for match in matches:
        home_team = match['homeTeam']['name']
        away_team = match['awayTeam']['name']
        home_goals = match['score']['fullTime']['home']
        away_goals = match['score']['fullTime']['away']
        
        if home_goals is None or away_goals is None:
            continue
            
        if home_team == team_name:
            goals_scored += home_goals
            goals_conceded += away_goals
            if home_goals > away_goals:
                wins += 1
            games_played += 1
            
        elif away_team == team_name:
            goals_scored += away_goals
            goals_conceded += home_goals
            if away_goals > home_goals:
                wins += 1
            games_played += 1
    
    if games_played == 0:
        return 0, 0, 0
    
    return (goals_scored / games_played, goals_conceded / games_played, wins)

def get_match_outcome(home_goals, away_goals):
    if home_goals > away_goals:
        return 2
    elif home_goals == away_goals:
        return 1
    else:
        return 0
    
    
    
    

def calculate_form_points(matches, team_name, last_n=5):
    points = 0
    games = 0

    for match in reversed(matches):
        if games >= last_n:
            break

        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]
        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]

        if home_goals is None or away_goals is None:
            continue

        if team_name != home_team and team_name != away_team:
            continue

        if home_goals == away_goals:
            points += 1
        else:
            team_won = (team_name == home_team and home_goals > away_goals) or \
                       (team_name == away_team and away_goals > home_goals)
            if team_won:
                points += 3

        games += 1

    return points

def calculate_home_away_stats(matches, team_name, is_home=True):
    goals_scored = 0
    goals_conceded = 0
    games_played = 0

    for match in matches:
        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]
        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]

        if home_goals is None or away_goals is None:
            continue

        if is_home and home_team == team_name:
            goals_scored += home_goals
            goals_conceded += away_goals
            games_played += 1

        elif not is_home and away_team == team_name:
            goals_scored += away_goals
            goals_conceded += home_goals
            games_played += 1

    if games_played == 0:
        return 0.0, 0.0

    return goals_scored / games_played, goals_conceded / games_played



# build training data
print("Building training data...")
training_data = []

for i, match in enumerate(matches):
    home_team = match["homeTeam"]["name"]
    away_team = match["awayTeam"]["name"]
    home_goals = match["score"]["fullTime"]["home"]
    away_goals = match["score"]["fullTime"]["away"]

    if home_goals is None or away_goals is None:
        continue

    previous_matches = matches[:i]

    if len(previous_matches) < 5:
        continue

    # Overall team stats
    home_goals_avg, home_conceded_avg, home_wins = calculate_team_stats(
        previous_matches, home_team
    )
    away_goals_avg, away_conceded_avg, away_wins = calculate_team_stats(
        previous_matches, away_team
    )

    # Home / Away specific stats
    home_home_goals_avg, home_home_conceded_avg = calculate_home_away_stats(
        previous_matches, home_team, is_home=True
    )
    away_away_goals_avg, away_away_conceded_avg = calculate_home_away_stats(
        previous_matches, away_team, is_home=False
    )

    # Feature engineering
    home_goal_diff = home_goals_avg - home_conceded_avg
    away_goal_diff = away_goals_avg - away_conceded_avg

    goal_diff_diff = home_goal_diff - away_goal_diff
    wins_diff = home_wins - away_wins

    home_form_points_5 = calculate_form_points(previous_matches, home_team, last_n=5)
    away_form_points_5 = calculate_form_points(previous_matches, away_team, last_n=5)

    features = {
        "home_goals_avg": home_goals_avg,
        "home_conceded_avg": home_conceded_avg,
        "home_wins": home_wins,
        "away_goals_avg": away_goals_avg,
        "away_conceded_avg": away_conceded_avg,
        "away_wins": away_wins,

        "home_goal_diff": home_goal_diff,
        "away_goal_diff": away_goal_diff,
        "goal_diff_diff": goal_diff_diff,
        "wins_diff": wins_diff,

        "home_form_points_5": home_form_points_5,
        "away_form_points_5": away_form_points_5,

        # NEW home/away features
        "home_home_goals_avg": home_home_goals_avg,
        "home_home_conceded_avg": home_home_conceded_avg,
        "away_away_goals_avg": away_away_goals_avg,
        "away_away_conceded_avg": away_away_conceded_avg,

        "outcome": get_match_outcome(home_goals, away_goals),
    }

    training_data.append(features)





df = pd.DataFrame(training_data)
print(f" Training samples: {len(df)}")

# Random forest model
print("\n Training Random Forest model...")

# Separate features (X) from outcome (y)
X = df.drop('outcome', axis=1)
y = df['outcome']

# Split into training and testing sets
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]

X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print(f" Training set: {len(X_train)} matches (earlier)")
print(f" Testing set: {len(X_test)} matches (later)")

print(f" Training set: {len(X_train)} matches")
print(f" Testing set: {len(X_test)} matches")

print("\nTrain outcome distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest outcome distribution:")
print(y_test.value_counts(normalize=True))


# Majority-class baseline: always predict the most common class in y_train
majority_class = int(y_train.value_counts().idxmax())
majority_preds = np.full(shape=len(y_test), fill_value=majority_class)
majority_acc = accuracy_score(y_test, majority_preds)

# Random-weighted baseline: sample using class distribution from y_train
class_probs = (y_train.value_counts(normalize=True)
               .reindex([0, 1, 2], fill_value=0.0)
               .values)

rng = np.random.default_rng(42)
random_preds = rng.choice([0, 1, 2], size=len(y_test), p=class_probs)
random_acc = accuracy_score(y_test, random_preds)

print("\n" + "="*50)
print("BASELINE RESULTS (time-based test set)")
print("="*50)
print(f"Majority-class baseline: {majority_acc * 100:.2f}% (predicts class {majority_class} always)")
print(f"Random-weighted baseline: {random_acc * 100:.2f}% (based on train distribution)")
print("="*50 + "\n")



# Create Random Forest with 100 trees
model = RandomForestClassifier(n_estimators=200, random_state=42,class_weight="balanced")

# Train the model
model.fit(X_train, y_train)

print(" Model trained!")

# Calculate accuracy
predictions = model.predict(X_test) 
accuracy = accuracy_score(y_test, predictions)

print(f" Model Accuracy: {accuracy * 100:.2f}%")
print(f"   ({int(accuracy * len(X_test))}/{len(X_test)} correct predictions)")

#  feature importance
print("\n Feature Importance:")
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in importance.iterrows():
    print(f"   {row['feature']:20s} {row['importance']*100:5.1f}%")



# Decision Tree Model

from sklearn.tree import DecisionTreeClassifier

print("\n Training Decision Tree model...")

# Create Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, class_weight="balanced")

# Train the model
dt_model.fit(X_train, y_train)

print(" Model trained!")

# Calculate accuracy
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print(f" Model Accuracy: {dt_accuracy * 100:.2f}%")
print(f"   ({int(dt_accuracy * len(X_test))}/{len(X_test)} correct predictions)")



# Logistic Regression Model

from sklearn.linear_model import LogisticRegression

print("\n Training Logistic Regression model...")

# Create Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=2000, class_weight="balanced")
# Train the model
lr_model.fit(X_train, y_train)

print(" Model trained!")

# Calculate accuracy
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print(f" Model Accuracy: {lr_accuracy * 100:.2f}%")
print(f"   ({int(lr_accuracy * len(X_test))}/{len(X_test)} correct predictions)")



# Model comparison

print("\n" + "="*50)
print("MODEL COMPARISON RESULTS")
print("="*50)
print(f"Random Forest:        {accuracy * 100:.2f}%")
print(f"Decision Tree:        {dt_accuracy * 100:.2f}%")
print(f"Logistic Regression:  {lr_accuracy * 100:.2f}%")
print("="*50)

# Determine best model
models = {
    'Random Forest': accuracy,
    'Decision Tree': dt_accuracy,
    'Logistic Regression': lr_accuracy
}
best_model = max(models, key=models.get)
print(f"\n Best performing model: {best_model} ({models[best_model] * 100:.2f}%)")

print("\nSaving trained models...")

# create folder if not exists
os.makedirs("models", exist_ok=True)

# Save each model
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/decision_tree.pkl", "wb") as f:
    pickle.dump(dt_model, f)

with open("models/logistic_regression.pkl", "wb") as f:
    pickle.dump(lr_model, f)

# Save results / best model info
model_results = {
    "random_forest": accuracy,
    "decision_tree": dt_accuracy,
    "logistic_regression": lr_accuracy,
    "best_model": best_model,
    "num_samples": len(df)
}

with open("models/model_results.pkl", "wb") as f:
    pickle.dump(model_results, f)

print(" Saved: models/random_forest.pkl")
print(" Saved: models/decision_tree.pkl")
print(" Saved: models/logistic_regression.pkl")
print(" Saved: models/model_results.pkl")