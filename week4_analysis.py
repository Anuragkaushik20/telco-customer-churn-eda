"""
Week 4 Task: Machine Learning Model Development and Evaluation
Author: Anurag Kaushik
Domain: Telco Customer Retention Analytics
Stack: Python 3.10+, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)

# Set visual styling
sns.set_theme(style="whitegrid", palette="deep")

def load_and_preprocess_data():
    """Load, clean, and prepare Telco Churn dataset for ML modeling."""
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    
    # Coerce TotalCharges and handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)
    
    # Define binary target
    df['Target'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Drop non-predictive identifiers
    X = df.drop(columns=['customerID', 'Churn', 'Target'])
    y = df['Target']
    
    return X, y

def build_preprocessing_pipeline(X):
    """Construct ColumnTransformer for numerical scaling and categorical encoding."""
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )
    return preprocessor, num_cols, cat_cols

def train_and_evaluate():
    """Train baseline & candidate models, print metrics, and save diagnostic plots."""
    X, y = load_and_preprocess_data()
    
    # Stratified 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    preprocessor, num_cols, cat_cols = build_preprocessing_pipeline(X)
    
    # 1. Baseline Model: Logistic Regression
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # 2. Candidate Model: Balanced Random Forest
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1
        ))
    ])
    
    # Fit models
    lr_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)
    
    # Generate predictions
    y_pred_lr = lr_pipeline.predict(X_test)
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    
    y_pred_rf = rf_pipeline.predict(X_test)
    y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    
    print("==========================================================================")
    print("          WEEK 4: MACHINE LEARNING MODEL BENCHMARKING REPORT          ")
    print("==========================================================================")
    
    models = {
        "Logistic Regression (Baseline)": (y_pred_lr, y_prob_lr),
        "Random Forest (Balanced Candidate)": (y_pred_rf, y_prob_rf)
    }
    
    for name, (preds, probs) in models.items():
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
        print(f"Precision: {precision_score(y_test, preds):.4f}")
        print(f"Recall:    {recall_score(y_test, preds):.4f}")
        print(f"F1-Score:  {f1_score(y_test, preds):.4f}")
        print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.4f}")
        print("\nClassification Report:\n", classification_report(y_test, preds))

    # --------------------------------------------------------------------
    # Plot 1: Normalized Confusion Matrices
    # --------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_lr = confusion_matrix(y_test, y_pred_lr, normalize='true')
    cm_rf = confusion_matrix(y_test, y_pred_rf, normalize='true')
    
    sns.heatmap(cm_lr, annot=True, fmt='.2%', cmap='Blues', ax=axes[0],
                xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
    axes[0].set_title('Baseline Logistic Regression\n(Normalized Confusion Matrix)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    sns.heatmap(cm_rf, annot=True, fmt='.2%', cmap='Oranges', ax=axes[1],
                xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
    axes[1].set_title('Cost-Weighted Random Forest\n(Normalized Confusion Matrix)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('eval_viz1_confusion_matrices.png', dpi=300)
    plt.close()
    
    # --------------------------------------------------------------------
    # Plot 2: ROC & Precision-Recall Curves
    # --------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # ROC Curve
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    
    axes[0].plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_lr):.3f})', color='#1f77b4', lw=2)
    axes[0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.3f})', color='#ff7f0e', lw=2)
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline')
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate (Recall)')
    axes[0].legend(loc='lower right')
    
    # PR Curve
    p_lr, r_lr, _ = precision_recall_curve(y_test, y_prob_lr)
    p_rf, r_rf, _ = precision_recall_curve(y_test, y_prob_rf)
    ap_lr = average_precision_score(y_test, y_prob_lr)
    ap_rf = average_precision_score(y_test, y_prob_rf)
    
    axes[1].plot(r_lr, p_lr, label=f'Logistic Regression (AP = {ap_lr:.3f})', color='#1f77b4', lw=2)
    axes[1].plot(r_rf, p_rf, label=f'Random Forest (AP = {ap_rf:.3f})', color='#ff7f0e', lw=2)
    axes[1].set_title('Precision-Recall (PR) Curve', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig('eval_viz2_roc_pr_curves.png', dpi=300)
    plt.close()
    
    # --------------------------------------------------------------------
    # Plot 3: Top Feature Importances
    # --------------------------------------------------------------------
    ohe_cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + ohe_cat_cols
    
    rf_model = rf_pipeline.named_steps['classifier']
    importances = rf_model.feature_importances_
    
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 5.5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='Viridis')
    plt.title('Top 10 Feature Importances (Balanced Random Forest)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Gini Importance Score', fontsize=11)
    plt.ylabel('Feature Name', fontsize=11)
    plt.tight_layout()
    plt.savefig('eval_viz3_feature_importance.png', dpi=300)
    plt.close()

    print("\nVisualizations successfully saved:")
    print(" - eval_viz1_confusion_matrices.png")
    print(" - eval_viz2_roc_pr_curves.png")
    print(" - eval_viz3_feature_importance.png")

if __name__ == '__main__':
    train_and_evaluate()
