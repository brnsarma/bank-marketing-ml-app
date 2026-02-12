"""
Utility functions for visualization and reporting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(model, X_test, y_test, model_name, ax=None):
    """
    Plot confusion matrix for a single model.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
           title=f'{model_name}\nAccuracy: {model.score(X_test, y_test):.3f}',
           ylabel='True Label', xlabel='Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    return ax

def create_performance_summary_table(results_df):
    """
    Create a styled performance summary table.
    """
    styled = results_df.style \
        .format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1 Score': '{:.4f}',
            'AUC Score': '{:.4f}',
            'MCC': '{:.4f}'
        }) \
        .background_gradient(subset=['Accuracy', 'F1 Score', 'AUC Score', 'MCC'], 
                           cmap='viridis') \
        .set_properties(**{'text-align': 'center'}) \
        .set_table_styles([{
            'selector': 'th',
            'props': [('background-color', '#2c3e50'), 
                     ('color', 'white'),
                     ('font-weight', 'bold')]
        }])
    return styled

def plot_model_comparison(results_df):
    """
    Create comparison visualizations for all models.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC Score', 'MCC']
    colors = sns.color_palette("viridis", len(results_df))
    
    for idx, metric in enumerate(metrics):
        if metric in results_df.columns:
            ax = axes[idx]
            bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig