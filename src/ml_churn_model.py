
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("Loading featured data...")
users = pd.read_csv('../data/users_featured.csv')

# --- Features & Target ---
FEATURES = ['total_sessions', 'avg_session_length', 'total_purchases',
            'total_levels', 'sessions_per_week',
            'F_score', 'M_score', 'RFM_score',
            'engagement_score', 'purchase_rate', 'level_rate',
            'is_high_spender', 'is_highly_active']

X = users[FEATURES]
y = users['churned']

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")

# --- Scale Features ---
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# --- Models ---
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, 
                              eval_metric='logloss', verbosity=0)
}

results = {}

print("\n" + "="*60)
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name == 'Logistic Regression':
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {'model': model, 'y_pred': y_pred, 
                     'y_prob': y_prob, 'f1': f1, 'auc': auc}
    
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

# --- ROC Curve Chart ---
plt.figure(figsize=(10, 6))
colors = ['#3498db', '#2ecc71', '#e74c3c']
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f"{name} (AUC = {res['auc']:.3f})")
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - All Models')
plt.legend()
plt.tight_layout()
plt.savefig('../data/chart8_roc_curves.png', dpi=150)
plt.close()
print("ROC curve saved!")

# --- Feature Importance (XGBoost) ---
xgb_model = results['XGBoost']['model']
importances = pd.Series(xgb_model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(10, 7))
importances.plot(kind='barh', color='#3498db')
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('../data/chart9_feature_importance.png', dpi=150)
plt.close()
print("Feature importance chart saved!")

# --- Confusion Matrix (Best Model) ---
best_name = max(results, key=lambda x: results[x]['auc'])
best_pred = results[best_name]['y_pred']
cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Churned','Churned'],
            yticklabels=['Not Churned','Churned'])
plt.title(f'Confusion Matrix - {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('../data/chart10_confusion_matrix.png', dpi=150)
plt.close()
print("Confusion matrix saved!")

# --- Summary ---
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<25} {'F1 Score':>10} {'ROC-AUC':>10}")
print("-"*45)
for name, res in results.items():
    print(f"{name:<25} {res['f1']:>10.4f} {res['auc']:>10.4f}")
print(f"\nBest Model: {best_name}")

# Save predictions
users_test = users.iloc[X_test.index].copy()
users_test['churn_probability'] = results[best_name]['y_prob']
users_test['predicted_churn'] = results[best_name]['y_pred']
users_test.to_csv('../data/users_with_predictions.csv', index=False)
print("Predictions saved to users_with_predictions.csv!")