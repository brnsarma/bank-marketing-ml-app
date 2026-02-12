"""
Data preprocessing module for Bank Marketing dataset.
Contains functions for loading, cleaning, encoding, and splitting data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(filepath=None):
    """
    Load Bank Marketing dataset from UCI or local file.
    """
    print("📂 Loading Bank Marketing dataset...")
    
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath, sep=';')
        print(f"✅ Loaded from local file: {df.shape}")
    else:
        # Fallback to ucimlrepo
        try:
            from ucimlrepo import fetch_ucirepo
            bank_marketing = fetch_ucirepo(id=222)
            df = bank_marketing.data.original
            print(f"✅ Loaded from UCI repository: {df.shape}")
        except:
            print("⚠️ Could not load from UCI, using sample data path")
            # If you have a local copy
            df = pd.read_csv('data/bank-additional-full.csv', sep=';')
    
    return df

def preprocess_data(df):
    """
    Complete preprocessing pipeline.
    """
    print("\n🔄 Starting preprocessing pipeline...")
    
    # Create a copy to avoid modifying original
    df_encoded = df.copy()
    
    # 1. Handle missing values
    print("   Step 1/5: Handling missing values...")
    
    # Check for missing values
    print("🔍 Checking for missing values...")
    missing_data = df_encoded.isnull().sum()
    missing_percentage = (df_encoded.isnull().sum() / len(df_encoded)) * 100

    missing_summary = pd.DataFrame({
        'Missing Values': missing_data,
        'Percentage (%)': missing_percentage.round(2)
    })

    # Filter to show only columns with missing values
    missing_cols = missing_summary[missing_summary['Missing Values'] > 0]

    if len(missing_cols) > 0:
        print(f"⚠️  Found {len(missing_cols)} columns with missing values:")
        print(missing_cols)
        
        # Handle missing values based on data type
        print("\n🛠️  Handling missing values...")
        
        for col in missing_cols.index:
            if df_encoded[col].dtype == 'object':  # Categorical
                mode_val = df_encoded[col].mode()[0]
                df_encoded[col].fillna(mode_val, inplace=True)
                print(f"   • {col}: Filled {missing_summary.loc[col, 'Missing Values']} missing values with mode '{mode_val}'")
            else:  # Numerical
                median_val = df_encoded[col].median()
                df_encoded[col].fillna(median_val, inplace=True)
                print(f"   • {col}: Filled {missing_summary.loc[col, 'Missing Values']} missing values with median {median_val:.2f}")
    else:
        print("✅ No missing values found in the dataset!")

    print("\n" + "=" * 70)
    
    # 2. Encode categorical variables
    print("   Step 2/5: Encoding categorical features...")
    print("=" * 70)

    # Identify categorical columns (excluding target)
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()

    if 'y' in categorical_cols:
        categorical_cols.remove('y')  # Handle target separately
        print(f"✅ Target variable 'y' will be encoded separately")

    print(f"📊 Found {len(categorical_cols)} categorical features:")
    for i, col in enumerate(categorical_cols, 1):
        unique_vals = df_encoded[col].nunique()
        print(f"   {i}. {col}: {unique_vals} unique values")

    # One-hot encode categorical variables (excluding target)
    if categorical_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
        print(f"✅ One-hot encoding completed for {len(categorical_cols)} features")

    # Encode target variable
    if 'y' in df_encoded.columns:
        target_mapping = {'no': 0, 'yes': 1}
        df_encoded['y'] = df_encoded['y'].map(target_mapping)
        print(f"✅ Target variable encoded: no→0, yes→1")

    print(f"   • Original shape: {df.shape}")
    print(f"   • After encoding: {df_encoded.shape}")
    
    # 3. Feature engineering
    print("\n" + "=" * 70)
    print("   Step 3/5: Creating new features...")
    print("=" * 70)

    print("🛠️  Creating new features...")

    # 1. Age groups
    df_encoded['age_group'] = pd.cut(df_encoded['age'], 
                                    bins=[0, 30, 40, 50, 60, 100],
                                    labels=['<30', '30-40', '40-50', '50-60', '60+'])

    # 2. Balance categories
    df_encoded['balance_category'] = pd.cut(df_encoded['balance'],
                                            bins=[-float('inf'), 0, 1000, 5000, float('inf')],
                                            labels=['negative', 'low', 'medium', 'high'])

    # 3. Campaign interaction rate (calls per duration)
    df_encoded['campaign_intensity'] = df_encoded['campaign'] / (df_encoded['duration'] + 1)

    # 4. Previous success indicator
    df_encoded['previous_success_ratio'] = df_encoded['pdays'].apply(
        lambda x: 1 if x == 999 else 0  # 999 means not previously contacted
    )

    print(f"✅ Created 4 new engineered features:")
    print("   1. age_group (categorical age ranges)")
    print("   2. balance_category (balance level categories)")
    print("   3. campaign_intensity (calls per duration)")
    print("   4. previous_success_ratio (previous contact success indicator)")

    # Convert new categorical features to one-hot
    new_categorical = ['age_group', 'balance_category']
    df_encoded = pd.get_dummies(df_encoded, columns=new_categorical, drop_first=True)

    print(f"📊 Final dataset shape after feature engineering: {df_encoded.shape}")
    
    # 4. Separate features and target
    print("\n" + "=" * 70)
    print("   Step 4/5: Separating X and y...")
    print("=" * 70)
    
    # Define target column
    target_col = 'y'
    
    # Separate features and target
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    
    print(f"✅ Features shape (X): {X.shape}")
    print(f"✅ Target shape (y): {y.shape}")
    print(f"✅ Target distribution:")
    print(f"   - Class 0 (No): {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"   - Class 1 (Yes): {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.1f}%)")
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    # 5. Train-test split
    print("\n" + "=" * 70)
    print("   Step 5/5: Creating train-test split...")
    print("=" * 70)

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Preserve class distribution
    )

    print(f"✅ Train-Test Split completed:")
    print(f"   • Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"   • Testing set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    # Scale features for models that need it
    print("\n🔄 Creating scaled versions for distance-based models...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✅ Feature scaling completed")
    print(f"   • Scaler mean shape: {scaler.mean_.shape}")
    print(f"   • Scaled train shape: {X_train_scaled.shape}")

    print("\n✅ Preprocessing pipeline complete!")
    
    # Return everything needed for training
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_names

def save_preprocessing_artifacts(scaler, feature_names, path='models/'):
    """
    Save scaler and feature names for later use in Streamlit.
    """
    os.makedirs(path, exist_ok=True)
    joblib.dump(scaler, f'{path}/scaler.pkl')
    joblib.dump(feature_names, f'{path}/feature_names.pkl')
    print(f"💾 Preprocessing artifacts saved to {path}")
    
    # Also save column info for reference
    column_info = {
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
        'scaler_scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
    }
    joblib.dump(column_info, f'{path}/preprocessing_info.pkl')
    print(f"💾 Preprocessing info saved to {path}")