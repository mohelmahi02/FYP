#train random forest model

import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#api data
USE_TEST_API = False
TEST_API_URL = "https://jsonblob.com/api/jsonBlob/019ae639-062a-7bc8-bd2f-65d55859bb27"

response = requests.get(TEST_API_URL)
data = response.json()
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

#build training data 
print("Building training data...")
training_data = []
for i, match in enumerate(matches):
    home_team = match['homeTeam']['name']
    away_team = match['awayTeam']['name']
    home_goals = match['score']['fullTime']['home']
    away_goals = match['score']['fullTime']['away']
    
    if home_goals is None or away_goals is None:
        continue
    
    previous_matches = matches[:i]
    
    if len(previous_matches) < 5:
        continue
    
    home_stats = calculate_team_stats(previous_matches, home_team)
    away_stats = calculate_team_stats(previous_matches, away_team)
    
    features = {
        'home_goals_avg': home_stats[0],
        'home_conceded_avg': home_stats[1],
        'home_wins': home_stats[2],
        'away_goals_avg': away_stats[0],
        'away_conceded_avg': away_stats[1],
        'away_wins': away_stats[2],
        'outcome': get_match_outcome(home_goals, away_goals)
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f" Training set: {len(X_train)} matches")
print(f" Testing set: {len(X_test)} matches")

# Create Random Forest with 100 trees
model = RandomForestClassifier(n_estimators=100, random_state=42)

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
dt_model = DecisionTreeClassifier(random_state=42)

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
lr_model = LogisticRegression(random_state=42, max_iter=1000)

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