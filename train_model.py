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

#Feature Columns
FEATURE_COLUMNS = [
    "HomeTeamShots",
    "AwayTeamShots",
    "HomeTeamShotsOnTarget",
    "AwayTeamShotsOnTarget",
    "HomeTeamCorners",
    "AwayTeamCorners",

    # Betting Odds (strong predictors)
    "B365HomeTeam",
    "B365Draw",
    "B365AwayTeam",
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
print("\nRandom Forest:")
print(pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False))

# Decision Tree
print("\nDecision Tree:")
print(pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": dt_model.feature_importances_
}).sort_values("importance", ascending=False))

# Logistic Regression
print("\nLogistic Regression:")
print(pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "coefficient": lr_model.coef_[0]
}).sort_values("coefficient", ascending=False))


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
