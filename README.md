# AI-Powered Premier League Match Predictor

A full-stack web application that uses machine learning to predict Premier League match outcomes. Built with Flask, React, and PostgreSQL, deployed on Render.com.

** Live App:** https://fyp-frontend-wgcl.onrender.com

** Screencast Demo: https://go.screenpal.com/watch/cOfOqKnOks3

** Current Performance:** 36.76% accuracy across 68 real-world predictions (GW27-30, 2025-26 season)

---

## Features

- **Match Predictions**: ML-powered predictions for upcoming Premier League fixtures
- **Live League Table**: Real-time standings from football-data.org API
- **Historical Performance**: Track prediction accuracy across multiple gameweeks
- **Model Comparison**: Compare Logistic Regression, Random Forest, and Decision Tree
- **16 Feature System**: Form metrics, goals, positions, and draw rates

---

## Tech Stack

### Backend
- **Python 3.12** - Core language
- **Flask** - REST API framework
- **PostgreSQL** - Database (Render managed)
- **scikit-learn** - Machine learning (Logistic Regression)
- **pandas** - Data processing
- **psycopg2** - PostgreSQL driver
- **python-dotenv** - Environment management

### Frontend
- **React 19** - UI framework
- **React Router** - Client-side routing
- **Tailwind CSS** - Styling
- **Axios** - HTTP client

### Deployment
- **Render.com** - Backend & database hosting
- **GitHub Actions** - CI/CD pipeline

---

## Installation

### Prerequisites
- Python 3.12+
- Node.js 18+
- PostgreSQL (optional - can use Render database)

### 1. Clone Repository
```bash
git clone https://github.com/mohelmahi02/FYP.git
cd FYP
```

### 2. Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt --break-system-packages

# Create .env file
cat > .env << 'ENVEOF'
DATABASE_URL=your_postgresql_url_here
FOOTBALL_DATA_API_KEY=your_api_key_here
ENVEOF
```

**Get API Key:** Sign up at https://www.football-data.org/client/register

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Database Initialization
```bash
# Run initialization script (uses DATABASE_URL from .env)
python init_render_db.py
```

---

## Running Locally

### Start Backend
```bash
# From project root
python app.py
# Backend runs at http://localhost:5000
```

### Start Frontend
```bash
# From frontend directory
npm start
# Frontend runs at http://localhost:3000
```

---

## Project Structure
```
FYP/
├── app.py                      # Flask API server
├── train_model.py              # ML model training
├── make_predictions.py         # Generate predictions
├── update_csv_with_current_season.py  # Fetch latest results
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in git)
│
├── services/
│   ├── db_service.py          # Database operations
│   ├── prediction_feature_service.py  # Feature engineering
│   ├── standings_service.py    # Live standings API
│   ├── compare_results.py      # Match predictions vs actuals
│   └── evaluation_service.py   # Calculate accuracy
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── decision_tree.pkl
│
├── data/
│   ├── premier_league.csv      # Historical match data (12,441 matches)
│   └── fixtures_gw24_38.csv    # 2025-26 season fixtures
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── Dashboard.jsx
    │   │   ├── History.jsx
    │   │   ├── ModelComparison.jsx
    │   │   └── TeamStats.jsx
    │   └── services/
    │       └── api.js
    └── package.json
```

---

## API Endpoints

### Backend (http://localhost:5000/api)
```
GET  /health              - Health check
GET  /models              - Model accuracies
GET  /predictions         - Upcoming predictions
GET  /history?limit=N     - Past predictions
GET  /accuracy            - Overall accuracy stats
GET  /standings           - Current Premier League table
```

---

## Machine Learning Pipeline

### Training
```bash
python train_model.py
```

**Process:**
1. Loads 2,261 matches from last 3 seasons (2023-2026)
2. Engineers 16 features per match
3. Trains 3 models (Logistic Regression, Random Forest, Decision Tree)
4. Saves best model (Logistic Regression: 59.74% training accuracy)

### Features (16 total)
1. HomeForm5 - Last 5 games points (home team)
2. AwayForm5 - Last 5 games points (away team)
3. HomeGoalsAvg - Average goals scored
4. AwayGoalsAvg - Average goals scored
5. HomeConcededAvg - Average goals conceded
6. AwayConcededAvg - Average goals conceded
7. FormCloseness - Form difference
8. GoalsCloseness - Goals difference
9. HomeDrawRate - Draw rate (last 5)
10. AwayDrawRate - Draw rate (last 5)
11. HomePosition - Form-based ranking
12. AwayPosition - Form-based ranking
13. PositionGap - Position difference
14. **HomeTablePos** - Actual league position (from API)
15. **AwayTablePos** - Actual league position (from API)
16. **TablePosGap** - Actual position difference

### Prediction
```bash
python make_predictions.py
```

**Process:**
1. Fetches next gameweek fixtures
2. Calculates 16 features per match
3. Generates probabilities (Home/Draw/Away)
4. Saves predictions to database

---

## Weekly Update Workflow

After each gameweek completes:
```bash
# 1. Update CSV with latest results
python update_csv_with_current_season.py

# 2. Compare predictions vs actual results
python -m services.compare_results

# 3. Check updated accuracy
python -m services.evaluation_service

# 4. Retrain model (optional)
python train_model.py

# 5. Generate next gameweek predictions
python make_predictions.py
```

---

## Real-World Performance

**Gameweek Results (2025-26 Season):**
- **GW27:** 30% (3/10 correct)
- **GW28:** 40% (4/10 correct)
- **GW29:** 10% (1/10 correct)
- **GW30:** 30% (3/10 correct)
- **GW31:** 50% (4/8 correct)
- **GW32:** 40% (4/10 correct)
- **GW33:** 60% (5/10 correct) — Best gameweek, exceeds industry standard
- 

**Overall: 36.76% (25/68 predictions)**

**Model Training Accuracy:** 59.74% (Logistic Regression)

**Industry Benchmarks:**
- Professional bookmakers: 50-55%
- Academic research: 40-50%
- This project: 36.76% (real-world)
-This project best gameweek: 60% (GW33)

---

## Environment Variables
```bash
# Required for backend
DATABASE_URL=postgresql://user:pass@host:5432/dbname
FOOTBALL_DATA_API_KEY=your_api_key_here
```

---

## Deployment

### Render.com Setup

**Backend Service:**
1. Create new Web Service
2. Connect GitHub repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Add environment variables

**Frontend (Static Site):**
1. Create Static Site
2. Build command: `cd frontend && npm install && npm run build`
3. Publish directory: `frontend/build`

---

## Future Improvements

- [ ] Add momentum/streak features
- [ ] Integrate injury data
- [ ] Ensemble model combining multiple algorithms
- [ ] Incorporate bookmaker odds as features
- [ ] Automate weekly prediction generation
- [ ] Add confidence-based filtering (show only >60% predictions)

---

## Author

**Mohammad Elmahi** - G00407950
BSc (Hons) Computing  
Atlantic Technological University, Galway-Mayo  

**Supervisor:** Gerard Harrison

**Academic Year:** 2025-2026

---

## License

This project is submitted as part of a Final Year Project at ATU Galway-Mayo.

---

## Acknowledgments

- **football-data.org** - Premier League data API
- **Kaggle** - Historical match dataset
- **Render.com** - Free hosting platform
- **Gerard Harrison** - Project supervision and guidance
