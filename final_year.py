#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)
from xgboost import XGBClassifier


# In[4]:


# ============================================================
# STEP 1 — Load Dataset
# ============================================================
print("=" * 60)
print("STEP 1: LOAD DATASET")
print("=" * 60)

df = pd.read_csv('alzheimers_disease_data.csv')  # change filename if needed

print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")


# In[5]:


# ============================================================
# STEP 2 — EDA (Exploratory Data Analysis)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: EDA")
print("=" * 60)

# Basic statistics
print("\n--- describe() ---")
print(df.describe().round(2))

# Missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Target column distribution
print("\n--- Target: Diagnosis (0 = No Alzheimer's, 1 = Alzheimer's) ---")
print(df['Diagnosis'].value_counts())
print(f"\nClass balance (%):\n{df['Diagnosis'].value_counts(normalize=True).round(3) * 100}")


# In[6]:


# ============================================================
# STEP 3 — Preprocessing
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: PREPROCESSING")
print("=" * 60)

# Drop PatientID and DoctorInCharge — these are just identifiers, not features
# They have no predictive value for Alzheimer's detection
drop_cols = [col for col in ['PatientID', 'DoctorInCharge'] if col in df.columns]
df.drop(columns=drop_cols, inplace=True)
print(f"Dropped columns: {drop_cols}")

# --- Label Encode any remaining object/categorical columns ---
# LabelEncoder converts text categories like "Male"/"Female" → 0/1
le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

if cat_cols:
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"Label encoded columns: {cat_cols}")
else:
    print("No categorical columns found — all columns are already numeric.")

# --- Fill missing values with column median ---
# Median is better than mean because it is not affected by outliers
missing_cols = df.columns[df.isnull().any()].tolist()
if missing_cols:
    for col in missing_cols:
        df[col].fillna(df[col].median(), inplace=True)
    print(f"Filled missing values in: {missing_cols}")
else:
    print("No missing values found.")

print(f"\nShape after preprocessing: {df.shape}")



# In[7]:


# ============================================================
# STEP 4 — Split Features and Target
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: SPLIT FEATURES (X) AND TARGET (y)")
print("=" * 60)

# X = all columns except Diagnosis
# y = Diagnosis column (0 or 1)
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Features used: {X.columns.tolist()}")


# In[8]:


# ============================================================
# STEP 5 — Train Test Split
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: TRAIN TEST SPLIT (80% train, 20% test)")
print("=" * 60)

# stratify=y ensures both train and test have same % of 0s and 1s
# Important for medical datasets where class imbalance can exist
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}  |  y_test: {y_test.shape}")


# In[9]:


# ============================================================
# STEP 6 — Feature Scaling (StandardScaler)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: FEATURE SCALING — StandardScaler")
print("=" * 60)

# StandardScaler makes every feature have mean=0 and std=1
# This is REQUIRED for Logistic Regression
# For tree-based models (Decision Tree, RF, XGBoost) it is not required
# but we apply it anyway for consistency
scaler = StandardScaler()

# fit_transform on train — learns mean & std from training data only
# transform on test — applies the same learned mean & std (no leakage)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Scaling done. Mean of each feature in X_train_scaled ≈ 0")
print(f"Sample means (first 5 features): {X_train_scaled.mean(axis=0)[:5].round(4)}")



# In[10]:


# ============================================================
# Helper function to print all metrics
# ============================================================
def evaluate_model(name, y_test, y_pred, y_prob=None):
    print(f"\n--- {name} ---")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    if y_prob is not None:
        print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")



# In[11]:


# ============================================================
# STEP 7 — Model 1: Logistic Regression
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: MODEL 1 — LOGISTIC REGRESSION")
print("=" * 60)

# Logistic Regression is the baseline model
# max_iter=1000 to ensure the solver converges
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]  # probability of class 1

evaluate_model("Logistic Regression", y_test, lr_pred, lr_prob)


# In[12]:


# ============================================================
# STEP 8 — Model 2: Decision Tree
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: MODEL 2 — DECISION TREE")
print("=" * 60)

# max_depth=5 prevents overfitting — tree won't grow too deep
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_scaled, y_train)

dt_pred = dt.predict(X_test_scaled)
dt_prob = dt.predict_proba(X_test_scaled)[:, 1]

evaluate_model("Decision Tree", y_test, dt_pred, dt_prob)



# In[13]:


# ============================================================
# STEP 9 — Model 3: Random Forest
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: MODEL 3 — RANDOM FOREST")
print("=" * 60)

# n_estimators=100 → builds 100 decision trees and averages them
# This reduces overfitting compared to a single Decision Tree
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

rf_pred = rf.predict(X_test_scaled)
rf_prob = rf.predict_proba(X_test_scaled)[:, 1]

evaluate_model("Random Forest", y_test, rf_pred, rf_prob)

# Feature importance — which features matter most for prediction
print("\nTop 10 Important Features (Random Forest):")
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
print(feat_imp.sort_values(ascending=False).head(10).round(4))


# In[14]:


# ============================================================
# STEP 10 — Model 4: XGBoost (Primary / Best Model)
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: MODEL 4 — XGBOOST (Primary Model)")
print("=" * 60)

# XGBoost is gradient boosting — builds trees sequentially
# Each new tree corrects errors made by the previous ones
# eval_metric='logloss' is standard for binary classification
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,       # step size for each boosting round
    max_depth=5,             # depth of each tree
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)
xgb.fit(X_train_scaled, y_train)

xgb_pred = xgb.predict(X_test_scaled)
xgb_prob = xgb.predict_proba(X_test_scaled)[:, 1]

evaluate_model("XGBoost", y_test, xgb_pred, xgb_prob)

# Feature importance for XGBoost
print("\nTop 10 Important Features (XGBoost):")
xgb_imp = pd.Series(xgb.feature_importances_, index=X.columns)
print(xgb_imp.sort_values(ascending=False).head(10).round(4))


# In[15]:


# ============================================================
# STEP 11 — Model Comparison Summary
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: MODEL COMPARISON SUMMARY")
print("=" * 60)

results = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'Accuracy': [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, xgb_pred)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, lr_prob),
        roc_auc_score(y_test, dt_prob),
        roc_auc_score(y_test, rf_prob),
        roc_auc_score(y_test, xgb_prob)
    ]
}

summary_df = pd.DataFrame(results)
summary_df['Accuracy'] = summary_df['Accuracy'].round(4)
summary_df['ROC-AUC']  = summary_df['ROC-AUC'].round(4)
print(summary_df.to_string(index=False))

best_model = summary_df.loc[summary_df['ROC-AUC'].idxmax(), 'Model']
print(f"\nBest model by ROC-AUC: {best_model}")

print("\n" + "=" * 60)
print("PROJECT COMPLETE")
print("=" * 60)


# In[16]:


# ============================================================
# STEP 12 — Use Best Model for Final Predictions
# ============================================================
print("\n" + "=" * 60)
print("STEP 12: BEST MODEL — FINAL PREDICTIONS")
print("=" * 60)

# Map model name → actual trained model object
model_map = {
    'Logistic Regression': lr,
    'Decision Tree':       dt,
    'Random Forest':       rf,
    'XGBoost':             xgb
}

# Pick the actual model object using the best model name
best_model = model_map[best_model]
print(f"Selected: {best_model}")

# Final predictions using the best model
final_pred = best_model.predict(X_test_scaled)
final_prob = best_model.predict_proba(X_test_scaled)[:, 1]

print(f"\nFinal Accuracy : {accuracy_score(y_test, final_pred):.4f}")
print(f"Final ROC-AUC  : {roc_auc_score(y_test, final_prob):.4f}")

print("\nFinal Confusion Matrix:")
print(confusion_matrix(y_test, final_pred))

print("\nFinal Classification Report:")
print(classification_report(y_test, final_pred,
      target_names=["No Alzheimer's (0)", "Alzheimer's (1)"]))

# Show sample predictions vs actual
print("Sample — Actual vs Predicted (first 10):")
sample = pd.DataFrame({
    'Actual'   : y_test.values[:10],
    'Predicted': final_pred[:10],
    "Probability (Alzheimer's)": final_prob[:10].round(3)
})
print(sample.to_string(index=False))

print("\n" + "=" * 60)
print("PROJECT COMPLETE")
print("=" * 60)

