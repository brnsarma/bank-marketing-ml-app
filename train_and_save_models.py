"""
Run this script once to train and save all models.
This script extracts the core logic from your notebook.
"""

import os
import sys
sys.path.append('.')

from src.data_preprocessing import load_data, preprocess_data, save_preprocessing_artifacts
from src.model_training import train_all_models, save_models
import joblib

def main():
    print("=" * 60)
    print("BANK MARKETING ML MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Load data
    df = load_data('data/bank-additional-full.csv')
    
    # 2. Preprocess
    # NEW CODE - expecting 8 values
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_names = preprocess_data(df)
    
    # 3. Save preprocessing artifacts
    save_preprocessing_artifacts(scaler, feature_names)
    
    # 4. Train models
    models, results_df = train_all_models(X_train, y_train, X_test, y_test)
    
    # 5. Save models
    save_models(models)
    
    # 6. Save results
    results_df.to_csv('models/model_performance.csv', index=False)
    print(f"💾 Results saved to models/model_performance.csv")
    
    print("\n✅ Training complete! Models are ready for Streamlit deployment.")
    print(f"📊 Best Model: {results_df.loc[results_df['F1 Score'].idxmax(), 'Model']}")
    print(f"   F1 Score: {results_df['F1 Score'].max():.4f}")
    
    return models, results_df

if __name__ == "__main__":
    models, results = main()