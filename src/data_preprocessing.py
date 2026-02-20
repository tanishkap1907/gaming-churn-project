import pandas as pd
import numpy as np

print("Loading data...")
sessions = pd.read_csv('../data/sessions_raw.csv')
users = pd.read_csv('../data/users_aggregated.csv')

print("\n--- SESSIONS DATA ---")
print(sessions.info())
print("\nMissing values:\n", sessions.isnull().sum())
print("\nDuplicates:", sessions.duplicated().sum())

print("\n--- USERS DATA ---")
print(users.info())
print("\nMissing values:\n", users.isnull().sum())

# --- Clean Sessions ---
# Drop duplicates
sessions = sessions.drop_duplicates()

# Fix date column
sessions['session_date'] = pd.to_datetime(sessions['session_date'])

# Remove outliers in session length (keep within 1st-99th percentile)
low = sessions['session_length_mins'].quantile(0.01)
high = sessions['session_length_mins'].quantile(0.99)
sessions = sessions[(sessions['session_length_mins'] >= low) & 
                    (sessions['session_length_mins'] <= high)]

# Encode categorical columns
sessions['device_type_encoded'] = sessions['device_type'].astype('category').cat.codes
sessions['region_encoded'] = sessions['region'].astype('category').cat.codes

print(f"\nSessions after cleaning: {sessions.shape}")

# --- Clean Users ---
users = users.drop_duplicates()

# Cap outliers in sessions_per_week
cap = users['sessions_per_week'].quantile(0.99)
users['sessions_per_week'] = users['sessions_per_week'].clip(upper=cap)

# Fill any nulls
users = users.fillna(0)

# Check class balance
print(f"\nChurn distribution:")
print(users['churned'].value_counts())
print(f"Churn rate: {users['churned'].mean():.1%}")

print("\n--- SAMPLE CLEANED USER DATA ---")
print(users.head())

# Save cleaned data
sessions.to_csv('../data/sessions_cleaned.csv', index=False)
users.to_csv('../data/users_cleaned.csv', index=False)

print("\nCleaned files saved!")