import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sessions = pd.read_csv('../data/sessions_cleaned.csv')
users = pd.read_csv('../data/users_cleaned.csv')

sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (10, 5)

print("Running EDA... charts will save to data/ folder")

# --- Chart 1: Churn Distribution ---
plt.figure()
ax = sns.countplot(x='churned', data=users, palette=['#2ecc71', '#e74c3c'])
ax.set_xticklabels(['Not Churned', 'Churned'])
ax.set_title('Churn Distribution')
ax.set_xlabel('')
ax.set_ylabel('Number of Users')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', 
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('../data/chart1_churn_distribution.png', dpi=150)
plt.close()
print("Chart 1 saved")

# --- Chart 2: Device Type Usage ---
plt.figure()
device_counts = sessions['device_type'].value_counts()
plt.pie(device_counts, labels=device_counts.index, autopct='%1.1f%%',
        colors=['#3498db','#e74c3c','#2ecc71','#f39c12'], startangle=140)
plt.title('Session Distribution by Device Type')
plt.tight_layout()
plt.savefig('../data/chart2_device_usage.png', dpi=150)
plt.close()
print("Chart 2 saved")

# --- Chart 3: Regional Activity ---
plt.figure()
region_counts = sessions['region'].value_counts()
sns.barplot(x=region_counts.values, y=region_counts.index, palette='Blues_r')
plt.title('Sessions by Region')
plt.xlabel('Number of Sessions')
plt.tight_layout()
plt.savefig('../data/chart3_regional_activity.png', dpi=150)
plt.close()
print("Chart 3 saved")

# --- Chart 4: Session Length Distribution ---
plt.figure()
sns.histplot(sessions['session_length_mins'], bins=50, color='#3498db', kde=True)
plt.title('Session Length Distribution')
plt.xlabel('Session Length (minutes)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('../data/chart4_session_length.png', dpi=150)
plt.close()
print("Chart 4 saved")

# --- Chart 5: Churn vs Avg Session Length ---
plt.figure()
sns.boxplot(x='churned', y='avg_session_length', data=users,
            palette=['#2ecc71','#e74c3c'])
plt.xticks([0,1], ['Not Churned','Churned'])
plt.title('Avg Session Length vs Churn')
plt.ylabel('Avg Session Length (mins)')
plt.tight_layout()
plt.savefig('../data/chart5_session_vs_churn.png', dpi=150)
plt.close()
print("Chart 5 saved")

# --- Chart 6: Correlation Heatmap ---
plt.figure(figsize=(10, 7))
cols = ['total_sessions','avg_session_length','total_purchases',
        'total_levels','days_since_last_login','sessions_per_week','churned']
corr = users[cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('../data/chart6_correlation_heatmap.png', dpi=150)
plt.close()
print("Chart 6 saved")

# --- Chart 7: Days Since Last Login vs Churn ---
plt.figure()
sns.boxplot(x='churned', y='days_since_last_login', data=users,
            palette=['#2ecc71','#e74c3c'])
plt.xticks([0,1], ['Not Churned','Churned'])
plt.title('Days Since Last Login vs Churn')
plt.ylabel('Days Since Last Login')
plt.tight_layout()
plt.savefig('../data/chart7_recency_vs_churn.png', dpi=150)
plt.close()
print("Chart 7 saved")

print("\nAll 7 charts saved to data/ folder!")
print("\nKey Insights:")
print(f"  Avg session length - Churned: {users[users.churned==1]['avg_session_length'].mean():.1f} mins")
print(f"  Avg session length - Active:  {users[users.churned==0]['avg_session_length'].mean():.1f} mins")
print(f"  Avg days since login - Churned: {users[users.churned==1]['days_since_last_login'].mean():.1f} days")
print(f"  Avg days since login - Active:  {users[users.churned==0]['days_since_last_login'].mean():.1f} days")
print(f"  Avg purchases - Churned: ${users[users.churned==1]['total_purchases'].mean():.2f}")
print(f"  Avg purchases - Active:  ${users[users.churned==0]['total_purchases'].mean():.2f}")