import os
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Load Data
df = pd.read_csv("data/premier_league.csv")
print("Total matches:", len(df))

#Target Variable
OUTCOME_MAP = {"A": 0, "D": 1, "H": 2}
df = df.dropna(subset=["FullTimeResult"])
df["outcome"] = df["FullTimeResult"].map(OUTCOME_MAP)

print("Adding fast form features...")
print("Adding goal strength features...")

df["HomeGoalsAvg"] = (
    df.groupby("HomeTeam")["FullTimeHomeTeamGoals"]
    .transform(lambda x: x.shift().rolling(5).mean())
)

df["AwayGoalsAvg"] = (
    df.groupby("AwayTeam")["FullTimeAwayTeamGoals"]
    .transform(lambda x: x.shift().rolling(5).mean())
)

df["HomeConcededAvg"] = (
    df.groupby("HomeTeam")["FullTimeAwayTeamGoals"]
    .transform(lambda x: x.shift().rolling(5).mean())
)

df["AwayConcededAvg"] = (
    df.groupby("AwayTeam")["FullTimeHomeTeamGoals"]
    .transform(lambda x: x.shift().rolling(5).mean())
)

df["HomeGoalDiff"] = df["HomeGoalsAvg"] - df["HomeConcededAvg"]
df["AwayGoalDiff"] = df["AwayGoalsAvg"] - df["AwayConcededAvg"]


# Sort by date first
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Store last results per team
team_history = {}

home_forms = []
away_forms = []

for _, row in df.iterrows():
    home = row["HomeTeam"]
    away = row["AwayTeam"]

    # Get last 5 games form points
    home_form = sum(team_history.get(home, [])[-5:])
    away_form = sum(team_history.get(away, [])[-5:])

    home_forms.append(home_form)
    away_forms.append(away_form)

    # Update history AFTER computing form
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

# Assign columns
df["HomeForm5"] = home_forms
df["AwayForm5"] = away_forms

#Feature Columns
FEATURE_COLUMNS = [
    "HomeForm5",
    "AwayForm5",
    "HomeGoalsAvg",
    "AwayGoalsAvg",
    "HomeGoalDiff",
    "AwayGoalDiff"
]




# Drop missing rows
df = df.dropna(subset=FEATURE_COLUMNS)

#Train-test split
train_df = df[df["Season"] < "2023-2024"]
test_df  = df[df["Season"] >= "2023-2024"]

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["outcome"]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df["outcome"]

print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))

#random forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

print("\nRandom Forest Accuracy:", rf_acc)


#Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_preds = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)

print("Decision Tree Accuracy:", dt_acc)

#Logistic Regression
lr_model = LogisticRegression(max_iter=2000)
lr_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)

print("Logistic Regression Accuracy:", lr_acc)

print("\n===============================")
print("FEATURE IMPORTANCE COMPARISON")
print("===============================")


# Random Forest 
rf_importance = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

rf_importance = rf_importance.reset_index(drop=True)
rf_importance.index += 1

print("\nRandom Forest (Most Important First):")
print(rf_importance)



# Decision Tree 
dt_importance = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": dt_model.feature_importances_
}).sort_values("importance", ascending=False)

dt_importance = dt_importance.reset_index(drop=True)
dt_importance.index += 1

print("\nDecision Tree (Most Important First):")
print(dt_importance)



# Logistic Regression 

lr_importance = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "coefficient": lr_model.coef_[0],
    "importance": abs(lr_model.coef_[0])
}).sort_values("importance", ascending=False)

lr_importance = lr_importance.reset_index(drop=True)
lr_importance.index += 1

print("\nLogistic Regression (Most Important First):")
print(lr_importance[["feature", "coefficient"]])


#Model Comparison
print("\n===============================")
print("MODEL COMPARISON")
print("===============================")
print(f"\nRandom Forest Accuracy: {rf_acc * 100:.2f}%")
print(f"Decision Tree Accuracy: {dt_acc * 100:.2f}%")
print(f"Logistic Regression Accuracy: {lr_acc * 100:.2f}%")
print("===============================")

# Best model
best_model = max(
    {"Random Forest": rf_acc, "Decision Tree": dt_acc, "Logistic Regression": lr_acc},
    key=lambda x: {"Random Forest": rf_acc, "Decision Tree": dt_acc, "Logistic Regression": lr_acc}[x]
)

print("Best Model:", best_model)

#Save models
os.makedirs("models", exist_ok=True)

with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("models/decision_tree.pkl", "wb") as f:
    pickle.dump(dt_model, f)

with open("models/logistic_regression.pkl", "wb") as f:
    pickle.dump(lr_model, f)

# Save results
model_results = {
    "random_forest": rf_acc,
    "decision_tree": dt_acc,
    "logistic_regression": lr_acc,
    "best_model": best_model,
    "num_samples": len(df)
}

with open("models/model_results.pkl", "wb") as f:
    pickle.dump(model_results, f)

print("\nModels saved successfully into /models/")
