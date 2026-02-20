import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print("Generating dataset... this may take 30-60 seconds")

np.random.seed(42)
NUM_SESSIONS = 1_000_000
NUM_USERS = 100_000

user_ids = np.random.randint(1, NUM_USERS + 1, size=NUM_SESSIONS)

devices = np.random.choice(
    ['Mobile', 'PC', 'Console', 'Tablet'],
    size=NUM_SESSIONS,
    p=[0.4, 0.3, 0.2, 0.1]
)

regions = np.random.choice(
    ['North America', 'Europe', 'Asia', 'South America', 'Others'],
    size=NUM_SESSIONS,
    p=[0.35, 0.25, 0.25, 0.10, 0.05]
)

session_lengths = np.random.exponential(scale=35, size=NUM_SESSIONS).clip(1, 300).round(1)

levels_completed = np.random.randint(0, 20, size=NUM_SESSIONS)

purchases = np.random.choice(
    [0, 0, 0, 0, 0, 1, 2, 5, 10, 20],
    size=NUM_SESSIONS
)

base_date = datetime(2024, 1, 1)
random_days = np.random.randint(0, 365, size=NUM_SESSIONS)
session_dates = [base_date + timedelta(days=int(d)) for d in random_days]

df = pd.DataFrame({
    'user_id': user_ids,
    'session_date': session_dates,
    'session_length_mins': session_lengths,
    'levels_completed': levels_completed,
    'in_game_purchases': purchases,
    'device_type': devices,
    'region': regions
})

df = df.sort_values(['user_id', 'session_date']).reset_index(drop=True)

reference_date = datetime(2024, 12, 31)
df['session_date'] = pd.to_datetime(df['session_date'])

user_agg = df.groupby('user_id').agg(
    total_sessions=('session_date', 'count'),
    avg_session_length=('session_length_mins', 'mean'),
    total_purchases=('in_game_purchases', 'sum'),
    total_levels=('levels_completed', 'sum'),
    last_login=('session_date', 'max'),
    first_login=('session_date', 'min')
).reset_index()

user_agg['days_since_last_login'] = (reference_date - user_agg['last_login']).dt.days
user_agg['days_active'] = (user_agg['last_login'] - user_agg['first_login']).dt.days + 1
user_agg['sessions_per_week'] = (user_agg['total_sessions'] / user_agg['days_active'] * 7).round(2)

# Churn label: inactive for 30+ days = churned
user_agg['churned'] = (user_agg['days_since_last_login'] >= 30).astype(int)

print(f"Sessions dataset: {df.shape}")
print(f"User dataset: {user_agg.shape}")
print(f"Churn rate: {user_agg['churned'].mean():.1%}")

df.to_csv('../data/sessions_raw.csv', index=False)
user_agg.to_csv('../data/users_aggregated.csv', index=False)

print("Done! Files saved to data/ folder")