"""
Model training module for all 6 classification models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, matthews_corrcoef)
import joblib
import os
import time

def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all 6 classification models and return results.
    """
    results = []
    models = {}
    
    # 1. Logistic Regression
    print("\n📊 Training Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced', solver='liblinear')
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    results.append(evaluate_model(lr, X_test, y_test, 'Logistic Regression'))
    
    # 2. Decision Tree
    print("📊 Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt
    results.append(evaluate_model(dt, X_test, y_test, 'Decision Tree'))
    
    # 3. KNN
    print("📊 Training K-Nearest Neighbors...")
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
    knn.fit(X_train, y_train)
    models['KNN'] = knn
    results.append(evaluate_model(knn, X_test, y_test, 'KNN'))
    
    # 4. Naive Bayes
    print("📊 Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    models['Naive Bayes'] = nb
    results.append(evaluate_model(nb, X_test, y_test, 'Naive Bayes'))
    
    # 5. Random Forest
    print("📊 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=15, 
                               random_state=42, class_weight='balanced_subsample')
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    results.append(evaluate_model(rf, X_test, y_test, 'Random Forest'))
    
    # 6. XGBoost
    print("📊 Training XGBoost...")
    xgb = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05,
                       random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    results.append(evaluate_model(xgb, X_test, y_test, 'XGBoost'))
    
    return models, pd.DataFrame(results)

def evaluate_model(model, X_test, y_test, name):
    """
    Evaluate a single model and return metrics dictionary.
    """
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred
    
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "AUC Score": roc_auc_score(y_test, y_prob),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

def save_models(models, path='models/'):
    """
    Save all trained models to disk.
    """
    os.makedirs(path, exist_ok=True)
    for name, model in models.items():
        filename = name.lower().replace(' ', '_').replace('-', '_') + '.pkl'
        joblib.dump(model, f'{path}/{filename}')
    print(f"💾 {len(models)} models saved to {path}")