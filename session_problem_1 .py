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
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    results[name] = roc_auc_score(y_test, y_proba)

# Best model
best_model = max(results, key=results.get)
print(f"\nBest Model: {best_model} with ROC-AUC: {results[best_model]:.4f}")

#OUTPUT 

============================================================
CREDIT CARD DEFAULT PREDICTION
============================================================
Dataset downloaded successfully!
Dataset shape: (30000, 25)

First 5 rows:
   ID  LIMIT_BAL  SEX  EDUCATION  MARRIAGE  AGE  PAY_0  PAY_2  PAY_3  PAY_4  \
0   1      20000    2          2         1   24      2      2     -1     -1   
1   2     120000    2          2         2   26     -1      2      0      0   
2   3      90000    2          2         2   34      0      0      0      0   
3   4      50000    2          2         1   37      0      0      0      0   
4   5      50000    1          2         1   57     -1      0     -1      0   

   ...  BILL_AMT4  BILL_AMT5  BILL_AMT6  PAY_AMT1  PAY_AMT2  PAY_AMT3  \
0  ...          0          0          0         0       689         0   
1  ...       3272       3455       3261         0      1000      1000   
2  ...      14331      14948      15549      1518      1500      1000   
3  ...      28314      28959      29547      2000      2019      1200   
4  ...      20940      19146      19131      2000     36681     10000   

   PAY_AMT4  PAY_AMT5  PAY_AMT6  default payment next month  
0         0         0         0                           1  
1      1000         0      2000                           1  
2      1000      1000      5000                           0  
3      1100      1069      1000                           0  
4      9000       689       679                           0  

[5 rows x 25 columns]

Column names: ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default_payment_next_month']

Target column: default_payment_next_month
Target distribution:
default_payment_next_month
0    23364
1     6636
Name: count, dtype: int64

After SMOTE - Class distribution: {1: 23364, 0: 23364}

🔵 Training Logistic Regression...
ROC-AUC Score: 0.7259
Classification Report:
              precision    recall  f1-score   support

           0       0.66      0.69      0.67      4664
           1       0.67      0.65      0.66      4682

    accuracy                           0.67      9346
   macro avg       0.67      0.67      0.67      9346
weighted avg       0.67      0.67      0.67      9346


🔵 Training Random Forest...
ROC-AUC Score: 0.9260
Classification Report:
              precision    recall  f1-score   support

           0       0.83      0.88      0.86      4664
           1       0.88      0.82      0.85      4682

    accuracy                           0.85      9346
   macro avg       0.85      0.85      0.85      9346
weighted avg       0.85      0.85      0.85      9346


🔵 Training XGBoost...
ROC-AUC Score: 0.9209
Classification Report:
              precision    recall  f1-score   support

           0       0.82      0.91      0.86      4664
           1       0.90      0.80      0.85      4682

    accuracy                           0.85      9346
   macro avg       0.86      0.85      0.85      9346
weighted avg       0.86      0.85      0.85      9346


Best Model: Random Forest with ROC-AUC: 0.9260
