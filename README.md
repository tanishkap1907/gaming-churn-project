
# 🎮 Cloud Gaming User Churn & Engagement Prediction

## Overview
End-to-end ML project analyzing 1M+ simulated gaming sessions to predict 
user churn and segment users by engagement behavior.

## Results
| Metric | Value |
|--------|-------|
| Dataset Size | 1M+ sessions, 99,990 users |
| Best Model | Logistic Regression |
| F1 Score | 0.93 |
| ROC-AUC | 0.986 |
| User Segments | 4 clusters |
| Churn Rate Identified | 45.1% |

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, Plotly Dash, Matplotlib, Seaborn

## Project Structure
- `src/` — Data generation, cleaning, EDA, ML models, clustering
- `data/` — Generated datasets and chart outputs  
- `dashboard/` — Interactive Plotly Dash app

## How to Run
pip install -r requirements.txt
cd src && python generate_data.py
cd src && python ml_churn_model.py
cd dashboard && python app.py