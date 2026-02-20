
import pandas as pd
import numpy as np

print("Loading cleaned user data...")
users = pd.read_csv('../data/users_cleaned.csv')

# --- RFM Features ---
# Recency: days since last login (already have it)
# Frequency: sessions per week (already have it)
# Monetary: total purchases (already have it)

# Normalize RFM into scores 1-5
def rfm_score(col, ascending=True):
    return pd.qcut(col, q=5, labels=[1,2,3,4,5], 
                   duplicates='drop').astype(float)

# Recency: lower days = better = higher score (ascending=False)
users['R_score'] = pd.qcut(users['days_since_last_login'], q=5,
                            labels=[5,4,3,2,1], duplicates='drop').astype(float)

# Frequency: more sessions = better
users['F_score'] = pd.qcut(users['sessions_per_week'], q=5,
                            labels=[1,2,3,4,5], duplicates='drop').astype(float)

# Monetary: more purchases = better
users['M_score'] = pd.qcut(users['total_purchases'].rank(method='first'), q=5,
                            labels=[1,2,3,4,5], duplicates='drop').astype(float)

# Combined RFM score
users['RFM_score'] = users['R_score'] + users['F_score'] + users['M_score']

# --- Engagement Score ---
users['engagement_score'] = (
    users['avg_session_length'] * 0.3 +
    users['sessions_per_week'] * 10 * 0.4 +
    users['total_levels'] / users['total_levels'].max() * 100 * 0.3
).round(2)

# --- Additional Features ---
# Purchase frequency (purchases per session)
users['purchase_rate'] = (users['total_purchases'] / users['total_sessions']).round(3)

# Level efficiency (levels per session)
users['level_rate'] = (users['total_levels'] / users['total_sessions']).round(3)

# Is high spender
users['is_high_spender'] = (users['total_purchases'] > users['total_purchases'].quantile(0.75)).astype(int)

# Is highly active
users['is_highly_active'] = (users['sessions_per_week'] > users['sessions_per_week'].quantile(0.75)).astype(int)

print("\nNew features added:")
print(users[['user_id','R_score','F_score','M_score','RFM_score',
             'engagement_score','purchase_rate','level_rate',
             'is_high_spender','is_highly_active','churned']].head(10))

print(f"\nFeature engineered dataset shape: {users.shape}")
print(f"\nRFM Score distribution:")
print(users['RFM_score'].describe())

print(f"\nEngagement Score distribution:")
print(users['engagement_score'].describe())

users.to_csv('../data/users_featured.csv', index=False)
print("\nFeature engineered data saved to users_featured.csv!")