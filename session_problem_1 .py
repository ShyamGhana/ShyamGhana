# Install required packages
!pip install -q pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost catboost shap joblib openpyxl kaggle

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import zipfile
import io
from google.colab import files
import requests
warnings.filterwarnings('ignore')

print("Setup complete! All packages installed.")

# ============================================
# PROBLEM 1: CREDIT CARD DEFAULT PREDICTION
# ============================================

print("="*60)
print("CREDIT CARD DEFAULT PREDICTION")
print("="*60)

# Method 1: Download directly from URL (no Kaggle account needed)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

try:
    df = pd.read_excel(url, header=1)
    print("✅ Dataset downloaded successfully!")
except:
    print("⚠️ Direct download failed. Upload your file manually.")
    from google.colab import files
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    df = pd.read_excel(filename, header=1)

print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Clean column names
df.columns = [col.strip().replace(' ', '_') for col in df.columns]
print("\nColumn names:", df.columns.tolist())

# Check target column
target_col = 'default_payment_next_month' if 'default_payment_next_month' in df.columns else df.columns[-1]
print(f"\nTarget column: {target_col}")
print(f"Target distribution:\n{df[target_col].value_counts()}")

# Basic preprocessing
X = df.drop([target_col, 'ID'] if 'ID' in df.columns else [target_col], axis=1)
y = df[target_col]

# Handle categorical variables
from sklearn.preprocessing import LabelEncoder
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

# Handle missing values
X = X.fillna(X.median())

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance with SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print(f"\nAfter SMOTE - Class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\n🔵 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    results[name] = roc_auc_score(y_test, y_proba)

# Best model
best_model = max(results, key=results.get)
print(f"\n✅ Best Model: {best_model} with ROC-AUC: {results[best_model]:.4f}")
