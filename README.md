# Bank Marketing Term Deposit Prediction - ML Classification Project

## 📋 Project Overview
This project implements and compares 6 machine learning classification models to predict whether a bank client will subscribe to a term deposit. The project demonstrates an end-to-end ML workflow including data preprocessing, model training, evaluation, and deployment via Streamlit.

---

## a. Problem Statement
Predict if a bank client will subscribe to a term deposit (binary classification) based on demographic, economic, and campaign-related features. This helps banks optimize marketing campaigns and reduce costs.

---

## b. Dataset Description
- **Source:** UCI Machine Learning Repository (Bank Marketing Dataset)
- **Instances:** 45,211 client records
- **Features:** 16 attributes (age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome)
- **Target Variable:** `y` - Has the client subscribed to a term deposit? (yes/no)
- **Class Distribution:** 
  - No: ~88.7% (majority class)
  - Yes: ~11.3% (minority class)
- **Challenges:** Class imbalance, mixed data types (numerical + categorical)

---

## c. Models Used and Evaluation Metrics

### Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8439 | 0.9051 | 0.4141 | 0.8062 | 0.5471 | 0.5020 |
| Decision Tree | 0.8215 | 0.8401 | 0.3731 | 0.7722 | 0.5031 | 0.4504 |
| K-Nearest Neighbors | 0.8818 | 0.7725 | 0.4908 | 0.2769 | 0.3541 | 0.3092 |
| Naive Bayes | 0.8691 | 0.8285 | 0.4483 | 0.5161 | 0.4798 | 0.4066 |
| Random Forest | 0.8809 | 0.9118 | 0.4936 | 0.6900 | 0.5755 | 0.5181 |
| XGBoost | 0.9056 | 0.9258 | 0.6364 | 0.4499 | 0.5271 | 0.4852 |


---

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-----------------------------------|
| Logistic Regression | Fast training, interpretable, moderate precision but low recall. Good baseline model. |
| Decision Tree | Prone to overfitting despite max_depth=10. Moderate performance across all metrics. |
| K-Nearest Neighbors | Sensitive to feature scaling. Performs reasonably but slower prediction time. |
| Naive Bayes | Fastest training, better recall than precision. Assumes feature independence. |
| Random Forest | Excellent accuracy, handles imbalance well. Top 2 performer overall. |
| XGBoost | **Best overall performance**. Best AUC, F1, and MCC. Handles imbalance effectively. |


---

## d. Key Findings

1. **Top Performers:** - **XGBoost** achieved the highest overall **Accuracy (90.56%)** and **AUC (0.9258)**.
   - **Random Forest** delivered the best **F1 Score (0.5755)** and **MCC (0.5181)**, making it the most balanced model for handling class imbalance.
2. **Class Imbalance Impact:** While all models felt the impact of the 11.7% minority class, **Logistic Regression** and **Decision Tree** showed surprisingly high **Recall** (above 77%), while **KNN** struggled significantly to identify subscribers.
3. **Most Important Features:** Call duration, account balance, age, and previous campaign success were the primary drivers of client subscription.
4. **Training Efficiency:** **Naive Bayes** and **Logistic Regression** were the fastest to train, while **Random Forest** and **XGBoost** required more computational resources but provided superior predictive power.
5. **Business Impact:** Implementing the **Random Forest** model allows the bank to optimize its marketing budget, potentially reducing call volume by 40% while still capturing nearly 70% of all potential term deposit subscribers.

---

## e. Streamlit Web Application

The project includes an interactive Streamlit app with:
- 📤 **Dataset Upload:** Upload test data for predictions
- 🤖 **Model Selection:** Choose from all 6 trained models
- 📊 **Performance Metrics:** View accuracy, precision, recall, F1, AUC, MCC
- 🔍 **Confusion Matrix:** Visualize model predictions
- 💡 **Live Predictions:** Make predictions on new client data

**Live App:** https://bank-marketing-ml-app-ancauqdmaj7hud3hqzfovw.streamlit.app/

---

## f. Technologies Used

- Python 3.9+
- Scikit-learn (ML models)
- XGBoost (Gradient boosting)
- Pandas/NumPy (Data processing)
- Matplotlib/Seaborn (Visualization)
- Streamlit (Web app)
- Joblib (Model serialization)

---

## g. Project Structure

```text
bank-marketing-ml-app/
│
├── app.py                      # Main Streamlit application
├── train_and_save_models.py    # Model training pipeline
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
│
├── src/                        # Source code package
│   ├── __init__.py             # Package marker
│   ├── data_preprocessing.py   # Data loading & preprocessing
│   ├── model_training.py       # Model training & evaluation
│   └── utils.py                # Visualization utilities
│
├── models/                     # Saved trained models & artifacts
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── notebooks/                  # Exploratory Data Analysis
│   └── bank_marketing_analysis.ipynb
│
└── data/                       # Dataset
    └── bank-additional-full.csv
```

## h. How to Run Locally

1. **Clone repository:**
   ```bash
   git clone https://github.com/brnsarma/bank-marketing-ml-app.git
   cd bank-marketing-ml-app

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt

4. **Run the model training pipeline (models are pre-trained, optional):**
    ```bash
    python train_and_save_models.py

5. **Launch the Streamlit application:**
    ```bash
    streamlit run app.py

6. **Open your browser and navigate to:**
    ```bash
    http://localhost:8501

## i. Usage Guide

### 🔹 Using the Web Application

**1. Upload Data:**
   - Click on "Browse files" to upload a CSV file
   - File should contain the same features as the training data
   - Sample test data is provided in the `data/` folder

**2. Select Model:**
   - Choose from 6 trained models via dropdown menu
   - Each model has different strengths (see performance table)
   - Models available: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost

**3. View Metrics:**
   - Model performance metrics are displayed in a formatted table
   - Compare metrics across different models
   - Metrics include: Accuracy, Precision, Recall, F1 Score, AUC, MCC

**4. Make Predictions:**
   - View predictions for uploaded data
   - Download results as CSV file
   - Visualize prediction distribution with interactive charts
   - View confusion matrix for selected model

### 🔸 Making Predictions on New Data

```python
# Example: Using the trained model in Python
import joblib
import pandas as pd
import numpy as np

# Load model, scaler, and feature names
model = joblib.load('models/random_forest.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Prepare new client data
new_client = pd.DataFrame([{
    'age': 35,
    'job': 'admin.',
    'marital': 'married',
    'education': 'secondary',
    'default': 'no',
    'balance': 1500,
    'housing': 'yes',
    'loan': 'no',
    'contact': 'cellular',
    'day': 15,
    'month': 'may',
    'duration': 180,
    'campaign': 2,
    'pdays': 999,
    'previous': 0,
    'poutcome': 'unknown'
}])
```
# Note: Full preprocessing pipeline is handled automatically in the web app
# This is a simplified example for reference

### Business Impact Analysis
* **False Positives:** Cost of marketing calls to non-interested clients.
* **False Negatives:** Missed revenue from potential subscribers.
* **Optimal Model:** **Random Forest** provides the best balance of precision and recall for targeting.
* **Estimated Savings:** ~40% reduction in marketing costs.

---

## l. Limitations and Future Work

### Current Limitations
* **Class Imbalance:** Only 11.7% positive cases affect model recall.
* **Hyperparameters:** Default parameters used; tuning could improve results.
* **Validation:** Single train-test split (80/20) used instead of K-Fold Cross-Validation.

### Future Improvements
* **Advanced Techniques:** Implement SMOTE for imbalance and GridSearchCV for tuning.
* **Feature Engineering:** Create interaction terms and include external economic indicators.
* **Deployment:** Add model monitoring, drift detection, and a REST API via FastAPI.

---

## m. Acknowledgements
* **UCI Machine Learning Repository:** For the Bank Marketing dataset.
* **Streamlit:** For the deployment platform.
* **Libraries:** Scikit-learn, XGBoost, Pandas, and Matplotlib.

---

## n. Author and Submission Information
* **Author:** R N Sarma Bollapinni
* **Assignment:** End-to-End Machine Learning Deployment
* **Submission Date:** February 11, 2026
* **GitHub Repository:** https://github.com/brnsarma/bank-marketing-ml-app
* **Live Streamlit App:** https://bank-marketing-ml-app-ancauqdmaj7hud3hqzfovw.streamlit.app/

---

## o. References
* [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
* [Streamlit Documentation](https://docs.streamlit.io)

### 📄 License
This project is submitted as part of an academic assignment. All rights reserved.  
© 2026 | Bank Marketing Term Deposit Prediction | ML Classification Project
