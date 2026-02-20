
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
users = pd.read_csv('../data/users_featured.csv')

CLUSTER_FEATURES = ['avg_session_length', 'sessions_per_week',
                    'total_purchases', 'total_levels', 
                    'engagement_score', 'purchase_rate']

X = users[CLUSTER_FEATURES].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Find optimal K using Elbow Method ---
print("Finding optimal number of clusters...")
inertias = []
sil_scores = []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
    print(f"  K={k} | Silhouette: {sil_scores[-1]:.4f}")

# --- Elbow Chart ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax2.plot(K_range, sil_scores, 'ro-')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score by K')
plt.tight_layout()
plt.savefig('../data/chart11_elbow_silhouette.png', dpi=150)
plt.close()
print("Elbow chart saved!")

# --- Final Model with K=4 ---
print("\nTraining final K=4 model...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
users['cluster'] = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, users['cluster'])
db = davies_bouldin_score(X_scaled, users['cluster'])
print(f"Silhouette Score: {sil:.4f}")
print(f"Davies-Bouldin Index: {db:.4f}")

# --- Analyze Clusters ---
cluster_summary = users.groupby('cluster').agg(
    user_count=('user_id', 'count'),
    avg_session_length=('avg_session_length', 'mean'),
    avg_sessions_per_week=('sessions_per_week', 'mean'),
    avg_purchases=('total_purchases', 'mean'),
    avg_engagement=('engagement_score', 'mean'),
    churn_rate=('churned', 'mean')
).round(2)

print("\nCluster Summary:")
print(cluster_summary)

# --- Label Clusters ---
def label_cluster(row):
    high_engage = row['avg_engagement'] >= cluster_summary['avg_engagement'].median()
    high_spend = row['avg_purchases'] >= cluster_summary['avg_purchases'].median()
    
    if high_engage and high_spend:
        return 'High Engagement'
    elif high_spend and not high_engage:
        return 'High Spender'
    elif high_engage and not high_spend:
        return 'Casual'
    else:
        return 'At-Risk'

cluster_summary['segment'] = cluster_summary.apply(label_cluster, axis=1)
print("\nCluster Labels:")
print(cluster_summary[['user_count','avg_engagement','churn_rate','segment']])

# Map labels back to users
label_map = cluster_summary['segment'].to_dict()
users['segment'] = users['cluster'].map(label_map)

# --- Cluster Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['#3498db','#2ecc71','#e74c3c','#f39c12']

# Plot 1: Cluster sizes
segment_counts = users['segment'].value_counts()
axes[0].pie(segment_counts, labels=segment_counts.index,
            autopct='%1.1f%%', colors=colors, startangle=140)
axes[0].set_title('User Segments Distribution')

# Plot 2: Engagement by segment
sns.boxplot(x='segment', y='engagement_score', data=users,
            ax=axes[1], palette=colors)
axes[1].set_title('Engagement Score by Segment')
axes[1].tick_params(axis='x', rotation=15)

# Plot 3: Churn rate by segment
churn_by_seg = users.groupby('segment')['churned'].mean().sort_values()
churn_by_seg.plot(kind='bar', ax=axes[2], color=colors)
axes[2].set_title('Churn Rate by Segment')
axes[2].set_ylabel('Churn Rate')
axes[2].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('../data/chart12_user_segments.png', dpi=150)
plt.close()
print("Segment charts saved!")

users.to_csv('../data/users_segmented.csv', index=False)
print("\nSegmented data saved!")

print("\n--- FINAL SEGMENT SUMMARY ---")
for cluster, row in cluster_summary.iterrows():
    print(f"\n{row['segment']} (Cluster {cluster})")
    print(f"  Users: {row['user_count']:,}")
    print(f"  Avg Engagement: {row['avg_engagement']}")
    print(f"  Churn Rate: {row['churn_rate']:.1%}")
    print(f"  Avg Purchases: ${row['avg_purchases']}")