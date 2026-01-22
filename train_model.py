#train random forest model

import os 
import pickle
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from services.feature_service import build_features


# real football-data.org API
API_KEY = "73dfd402f27440d4aff1f6d50185fb3a"
API_URL = "https://api.football-data.org/v4/competitions/PL/matches"

headers = {"X-Auth-Token": API_KEY}
response = requests.get(API_URL, headers=headers)
data = response.json()

print(f"Fetched {len(data['matches'])} matches from API")

matches = data['matches']

def get_match_outcome(home_goals, away_goals):
    """
    Encode match outcome:
    2 = Home Win
    1 = Draw
    0 = Away Win
    """
    if home_goals > away_goals:
        return 2
    elif home_goals == away_goals:
        return 1
    else:
        return 0





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

    X_row = build_features(previous_matches, home_team, away_team)
    X_row["outcome"] = get_match_outcome(home_goals, away_goals)

    training_data.append(X_row.iloc[0].to_dict())






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