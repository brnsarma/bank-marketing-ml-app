"""
Streamlit Web Application for Bank Marketing Term Deposit Prediction
Author: R N Sarma Bollapinni
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Bank Marketing ML App",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #34495e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND ARTIFACTS
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models and preprocessing artifacts."""
    models = {}
    model_files = {
        'Logistic Regression': 'models/logistic_regression.pkl',
        'Decision Tree': 'models/decision_tree.pkl',
        'K-Nearest Neighbors': 'models/knn.pkl',
        'Naive Bayes': 'models/naive_bayes.pkl',
        'Random Forest': 'models/random_forest.pkl',
        'XGBoost': 'models/xgboost.pkl'
    }
    
    for name, path in model_files.items():
        try:
            models[name] = joblib.load(path)
        except FileNotFoundError:
            st.error(f"❌ Model file not found: {path}")
            st.info("Please run train_and_save_models.py first to train and save the models.")
            st.stop()
    
    # Load preprocessing artifacts
    try:
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
    except FileNotFoundError:
        st.error("❌ Preprocessing artifacts not found.")
        st.info("Please run train_and_save_models.py first.")
        st.stop()
    
    return models, scaler, feature_names

@st.cache_data
def load_performance_metrics():
    """
    Load model performance metrics with FIXED, STANDARDIZED model names.
    This ensures dropdown and metrics names ALWAYS match.
    """
    
    # === STANDARD MODEL NAMES ===
    STANDARD_MODEL_NAMES = [
        'Logistic Regression',
        'Decision Tree', 
        'K-Nearest Neighbors',
        'Naive Bayes',
        'Random Forest',
        'XGBoost'
    ]
    
   
    METRICS_DATA = {
        'Accuracy': [0.8439, 0.8215, 0.8818, 0.8691, 0.8809, 0.9056],
        'Precision': [0.4141, 0.3731, 0.4908, 0.4483, 0.4936, 0.6364],
        'Recall': [0.8062, 0.7722, 0.2769, 0.5161, 0.6900, 0.4499],
        'F1 Score': [0.5471, 0.5031, 0.3541, 0.4798, 0.5755, 0.5271],
        'AUC Score': [0.9051, 0.8401, 0.7725, 0.8285, 0.9118, 0.9258],
        'MCC': [0.5020, 0.4504, 0.3092, 0.4066, 0.5181, 0.4852]
    }
    
    # === CREATE DATAFRAME WITH STANDARD NAMES ===
    metrics_df = pd.DataFrame(METRICS_DATA)
    metrics_df.insert(0, 'Model', STANDARD_MODEL_NAMES)
    
    return metrics_df

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_input_data(df, feature_names):
    """
    Properly preprocess uploaded data to match training features.
    Handles categorical encoding and feature engineering EXACTLY like training.
    """
    df_processed = df.copy()
    
    # === 1. ENCODE TARGET (if present) ===
    if 'y' in df_processed.columns:
        df_processed['y'] = df_processed['y'].map({'no': 0, 'yes': 1})
    
    # === 2. IDENTIFY CATEGORICAL COLUMNS ===
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                        'loan', 'contact', 'month', 'poutcome']
    
    # === 3. ONE-HOT ENCODE CATEGORICAL VARIABLES ===
    for col in categorical_cols:
        if col in df_processed.columns:
            # Get dummies with drop_first=True (same as training)
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            df_processed.drop(col, axis=1, inplace=True)
            print(f"✅ Encoded {col} -> {len(dummies.columns)} columns")  # Debug
    
    # === 4. FEATURE ENGINEERING ===
    # Age groups
    if 'age' in df_processed.columns:
        df_processed['age_group'] = pd.cut(df_processed['age'], 
                                          bins=[0, 30, 40, 50, 60, 100],
                                          labels=['<30', '30-40', '40-50', '50-60', '60+'])
        age_dummies = pd.get_dummies(df_processed['age_group'], prefix='age_group', drop_first=True)
        df_processed = pd.concat([df_processed, age_dummies], axis=1)
        df_processed.drop('age_group', axis=1, inplace=True)
    
    # Balance categories
    if 'balance' in df_processed.columns:
        df_processed['balance_category'] = pd.cut(df_processed['balance'],
                                                 bins=[-float('inf'), 0, 1000, 5000, float('inf')],
                                                 labels=['negative', 'low', 'medium', 'high'])
        balance_dummies = pd.get_dummies(df_processed['balance_category'], 
                                        prefix='balance_category', drop_first=True)
        df_processed = pd.concat([df_processed, balance_dummies], axis=1)
        df_processed.drop('balance_category', axis=1, inplace=True)
    
    # Campaign intensity
    if 'campaign' in df_processed.columns and 'duration' in df_processed.columns:
        df_processed['campaign_intensity'] = df_processed['campaign'] / (df_processed['duration'] + 1)
    
    # Previous success ratio
    if 'pdays' in df_processed.columns:
        df_processed['previous_success_ratio'] = df_processed['pdays'].apply(
            lambda x: 1 if x == 999 else 0
        )
    
    # === 5. ENSURE ALL TRAINING FEATURES EXIST ===
    for col in feature_names:
        if col not in df_processed.columns:
            df_processed[col] = 0  # Add missing columns with 0
    
    # === 6. KEEP ONLY TRAINING FEATURES IN CORRECT ORDER ===
    df_processed = df_processed[feature_names]
    
    print(f"✅ Final processed shape: {df_processed.shape}")
    print(f"✅ Expected features: {len(feature_names)}")
    
    return df_processed

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix using seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold')
    return fig

def plot_roc_curve(y_true, y_prob, model_name):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title(f'ROC Curve - {model_name}', fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Bank Marketing Term Deposit Prediction</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; background-color: #e8f4f8; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; color: #2c3e50; margin: 0;'>
            Predict whether a client will subscribe to a term deposit using 6 different ML models
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models and artifacts
    with st.spinner('🔄 Loading models and preprocessing artifacts...'):
        models, scaler, feature_names = load_models()
        metrics_df = load_performance_metrics()
    st.success('✅ Models loaded successfully!')
    
    # ========================================================================
    # SIDEBAR - MODEL SELECTION AND UPLOAD
    # ========================================================================
    
    with st.sidebar:
        st.markdown('<h2 style="color: #2c3e50;">⚙️ Controls</h2>', unsafe_allow_html=True)
        
        # Model selection
        st.markdown('<h3 style="color: #34495e;">🤖 Select Model</h3>', unsafe_allow_html=True)
        selected_model = st.selectbox(
            "Choose a classification model:",
            list(models.keys()),
            index=4  # Default to Random Forest
        )
        
        st.markdown("---")
        
        # Dataset upload - Step 6a: Dataset upload option
        st.markdown('<h3 style="color: #34495e;">📤 Upload Test Data</h3>', unsafe_allow_html=True)
        st.markdown("*Streamlit free tier has limited capacity - upload only test data*")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="File should contain the same features as the training data"
        )
        
        # Sample data option
        use_sample = st.checkbox("Use sample test data", value=True)
        
        st.markdown("---")
        
        # Prediction button
        predict_button = st.button("🚀 Make Predictions", use_container_width=True)
        
        st.markdown("---")
        
        # Model info
        st.markdown('<h3 style="color: #34495e;">ℹ️ Model Info</h3>', unsafe_allow_html=True)
        if selected_model == 'Random Forest':
            st.info("🏆 **Best F1 Score:** 0.5755\n\n✅ Handles imbalance well\n\n🌳 Ensemble method")
        elif selected_model == 'XGBoost':
            st.info("⚡ **Best Accuracy**\n\n📈 High AUC score\n\n🚀 Gradient boosting")
        elif selected_model == 'Logistic Regression':
            st.info("📊 Good baseline\n\n⚡ Fast training\n\n📋 Interpretable")
        else:
            st.info(f"ℹ️ Selected: {selected_model}")
    
    # ========================================================================
    # MAIN CONTENT - METRICS AND PREDICTIONS
    # ========================================================================
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Step 6c: Display evaluation metrics
        st.markdown('<h2 class="sub-header">📊 Model Performance Metrics</h2>', unsafe_allow_html=True)
        
        # Display metrics for selected model
        model_metrics = metrics_df[metrics_df['Model'] == selected_model].iloc[0]
        
        # Create metrics cards
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        
        with mcol1:
            st.metric(
                label="🎯 Accuracy",
                value=f"{model_metrics['Accuracy']:.4f}",
                delta=f"{model_metrics['Accuracy'] - 0.8:.4f}" if model_metrics['Accuracy'] > 0.8 else None
            )
            st.metric(
                label="📏 Precision",
                value=f"{model_metrics['Precision']:.4f}"
            )
        
        with mcol2:
            st.metric(
                label="🎯 Recall",
                value=f"{model_metrics['Recall']:.4f}"
            )
            st.metric(
                label="⚖️ F1 Score",
                value=f"{model_metrics['F1 Score']:.4f}",
                delta=f"{model_metrics['F1 Score'] - 0.5:.4f}" if model_metrics['F1 Score'] > 0.5 else None
            )
        
        with mcol3:
            st.metric(
                label="📈 AUC Score",
                value=f"{model_metrics['AUC Score']:.4f}"
            )
            st.metric(
                label="🔷 MCC",
                value=f"{model_metrics['MCC']:.4f}"
            )
        
        with mcol4:
            # Model comparison summary
            best_model = metrics_df.loc[metrics_df['F1 Score'].idxmax(), 'Model']
            st.info(f"🏆 **Best Model:**\n{best_model}")
            st.info(f"📊 **Your selection:**\n{selected_model}")
    
    with col2:
        # Model comparison chart
        st.markdown('<h3 style="color: #34495e;">📈 Model Comparison</h3>', unsafe_allow_html=True)
        
        fig = go.Figure(data=[
            go.Bar(name='F1 Score', x=metrics_df['Model'], y=metrics_df['F1 Score'], marker_color='#3498db'),
            go.Bar(name='AUC', x=metrics_df['Model'], y=metrics_df['AUC Score'], marker_color='#2ecc71')
        ])
        fig.update_layout(
            barmode='group',
            height=300,
            margin=dict(l=20, r=20, t=30, b=80),
            xaxis_tickangle=-45,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PREDICTIONS SECTION
    # ========================================================================
    
    st.markdown('<h2 class="sub-header">🔮 Predictions</h2>', unsafe_allow_html=True)
    
    # Load data
    if use_sample:
        # Create sample test data
        sample_data = pd.DataFrame({
            'age': [35, 42, 28, 55, 31],
            'balance': [1500, 3200, 800, 12500, 2100],
            'duration': [180, 320, 95, 450, 230],
            'campaign': [2, 1, 3, 1, 2],
            'pdays': [999, 999, 180, 999, 45],
            'previous': [0, 0, 1, 0, 2]
        })
        df = sample_data
        st.info("📋 Using sample test data. Upload your own CSV file to make predictions on custom data.")
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} records from {uploaded_file.name}")
    else:
        df = None
        st.warning("⚠️ Please upload a CSV file or use sample data to make predictions.")
    
    if df is not None and predict_button:
        with st.spinner(f'🔄 Making predictions using {selected_model}...'):
            try:
                # Preprocess data
                df_processed = preprocess_input_data(df, feature_names)
                
                # Scale features
                df_scaled = scaler.transform(df_processed)
                
                # Get model
                model = models[selected_model]
                
                # Make predictions
                predictions = model.predict(df_scaled)
                probabilities = model.predict_proba(df_scaled)[:, 1]
                
                # Create results dataframe
                results_df = df.copy()
                results_df['Prediction'] = predictions
                results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'No', 1: 'Yes'})
                results_df['Probability'] = probabilities
                results_df['Confidence'] = np.where(
                    results_df['Prediction'] == 1,
                    results_df['Probability'],
                    1 - results_df['Probability']
                )
                
                # Display results
                st.markdown('<h3 style="color: #34495e;">📋 Prediction Results</h3>', unsafe_allow_html=True)
                
                # Summary stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Predictions", len(results_df))
                with col2:
                    st.metric("Predicted Yes", (results_df['Prediction'] == 1).sum())
                with col3:
                    st.metric("Predicted No", (results_df['Prediction'] == 0).sum())
                with col4:
                    st.metric("Avg Confidence", f"{results_df['Confidence'].mean():.2%}")
                
                # Show results table
                st.dataframe(
                    results_df.style.applymap(
                        lambda x: 'background-color: #d4edda' if x == 'Yes' else 
                                 ('background-color: #f8d7da' if x == 'No' else ''),
                        subset=['Prediction_Label']
                    ),
                    use_container_width=True
                )
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"{selected_model}_predictions.csv",
                    mime="text/csv",
                )
                
                # Visualizations
                st.markdown('<h3 style="color: #34495e;">📊 Prediction Distribution</h3>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig = px.pie(
                        results_df, 
                        names='Prediction_Label',
                        title='Prediction Distribution',
                        color_discrete_sequence=['#2ecc71', '#e74c3c'],
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Probability distribution
                    fig = px.histogram(
                        results_df,
                        x='Probability',
                        nbins=20,
                        title='Probability Distribution',
                        labels={'Probability': 'Prediction Probability'},
                        color_discrete_sequence=['#3498db']
                    )
                    fig.add_vline(x=0.5, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confusion Matrix & Classification Report - Step 6d
                st.markdown("---")
                st.markdown('<h2 class="sub-header">🔍 Model Evaluation</h2>', unsafe_allow_html=True)
                
                # Check if the uploaded data contains the true labels (column 'y')
                if 'y' in df.columns:
                    st.success("✅ True labels found! Showing confusion matrix and classification report.")
                    
                    # Convert true labels to numeric
                    y_true = df['y'].map({'no': 0, 'yes': 1})
                    y_pred = predictions
                    
                    # Create two columns for confusion matrix and classification report
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Confusion Matrix")
                        # Plot confusion matrix
                        cm = confusion_matrix(y_true, y_pred)
                        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                                   annot_kws={'size': 14})
                        ax_cm.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
                        ax_cm.set_ylabel('True Label', fontweight='bold', fontsize=12)
                        ax_cm.set_title(f'Confusion Matrix - {selected_model}', fontweight='bold', fontsize=14)
                        st.pyplot(fig_cm)
                        plt.close()
                    
                    with col2:
                        st.markdown("### 📋 Classification Report")
                        # Generate classification report
                        report = classification_report(y_true, y_pred, 
                                                      target_names=['No', 'Yes'],
                                                      output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        
                        # Format and display
                        st.dataframe(
                            report_df.style.format("{:.3f}")
                            .background_gradient(cmap='viridis', subset=pd.IndexSlice['No':'Yes', :]),
                            use_container_width=True
                        )
                        
                        # Calculate and display additional metrics
                        tn, fp, fn, tp = cm.ravel()
                        st.markdown("### 📈 Additional Metrics")
                        metrics_data = {
                            'Metric': ['True Negatives', 'False Positives', 'False Negatives', 'True Positives', 
                                      'Sensitivity (Recall)', 'Specificity', 'Precision', 'F1 Score'],
                            'Value': [tn, fp, fn, tp, 
                                     tp/(tp+fn), tn/(tn+fp), 
                                     tp/(tp+fp), 2*tp/(2*tp+fp+fn)]
                        }
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df.style.format({'Value': '{:.0f}' if isinstance(metrics_df['Value'].iloc[0], int) else '{:.3f}'}),
                                    use_container_width=True)
                
                else:
                    # If no true labels, show message and option to download predictions
                    st.info("ℹ️ To see confusion matrix and classification report, upload a CSV file with a 'y' column containing true labels (yes/no).")
                    
                    # Example format
                    with st.expander("📋 View expected CSV format for evaluation"):
                        example_data = pd.DataFrame({
                            'age': [35, 42, 28],
                            'job': ['admin.', 'technician', 'services'],
                            'marital': ['married', 'married', 'single'],
                            'education': ['secondary', 'tertiary', 'secondary'],
                            'y': ['no', 'no', 'yes']  # Include true labels
                        })
                        st.dataframe(example_data)
                        st.caption("Upload a CSV with ALL features plus a 'y' column for evaluation")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
        <p>🏦 Bank Marketing Term Deposit Prediction | End-to-End ML Deployment</p>
        <p style='font-size: 0.9rem;'>Built with Streamlit • Scikit-learn • XGBoost</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
