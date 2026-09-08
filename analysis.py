"""
Week 1 Task: Data Acquisition, Cleaning, and Exploratory Data Analysis (EDA)
Author: Data Science Intern
Domain: Telco Customer Retention Analytics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot styling
sns.set_theme(style="whitegrid", palette="deep")

def load_data():
    """Acquire public dataset from repository."""
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    print(f"Ingesting data from: {url}")
    df = pd.read_csv(url)
    print(f"Raw Data Ingested: {df.shape[0]} rows x {df.shape[1]} columns")
    return df

def clean_data(df):
    """Handle latent nulls, whitespace corruptions, and type mismatches."""
    print("\n--- Running Data Cleaning Pipeline ---")
    
    # Check for duplicate customer IDs
    duplicates = df['customerID'].duplicated().sum()
    print(f"Duplicate Customer IDs found: {duplicates}")
    
    # Coerce whitespace TotalCharges to float; fill zero-tenure rows with 0.0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    latent_nulls = df['TotalCharges'].isnull().sum()
    print(f"Detected {latent_nulls} blank TotalCharges records (tenure = 0). Imputing with 0.0.")
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # Target encoding for correlation analysis
    df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    assert df.isnull().sum().sum() == 0, "Null values remain unresolved."
    print("Data cleaning completed successfully. 0 null values remaining.\n")
    return df

def generate_visualizations(df):
    """Generate and save the 3 required exploratory charts."""
    print("--- Generating Visualizations ---")
    
    # Visualization 1: Attrition Rate by Contract Horizon
    plt.figure(figsize=(8, 5))
    contract_churn = df.groupby('Contract')['Churn_Numeric'].agg(['mean', 'count']).reset_index()
    contract_churn['Percentage'] = contract_churn['mean'] * 100
    
    ax = sns.barplot(x='Contract', y='Percentage', data=contract_churn, palette='Blues_r')
    plt.title('Customer Churn Rate by Contract Commitment Type', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Contract Type', fontsize=11)
    plt.ylabel('Attrition Rate (%)', fontsize=11)
    plt.ylim(0, 50)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('viz1_contract_churn_rate.png', dpi=300)
    plt.close()
    print("Saved: viz1_contract_churn_rate.png")

    # Visualization 2: Tenure Density Distribution
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df[df['Churn'] == 'Yes']['tenure'], label='Churned Customers', fill=True, color='#e53e3e', alpha=0.4)
    sns.kdeplot(data=df[df['Churn'] == 'No']['tenure'], label='Retained Customers', fill=True, color='#3182ce', alpha=0.4)
    plt.title('Subscriber Tenure Density: Retained vs. Churned Cohorts', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Account Tenure (Months)', fontsize=11)
    plt.ylabel('Density Distribution', fontsize=11)
    plt.legend(title='Status', loc='upper right')
    plt.xlim(0, 72)
    plt.tight_layout()
    plt.savefig('viz2_tenure_density_distribution.png', dpi=300)
    plt.close()
    print("Saved: viz2_tenure_density_distribution.png")

    # Visualization 3: Correlation Heatmap
    plt.figure(figsize=(7, 6))
    numeric_fields = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_Numeric']
    corr_matrix = df[numeric_fields].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1.0, vmax=1.0, linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Metrics and Churn', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('viz3_correlation_heatmap.png', dpi=300)
    plt.close()
    print("Saved: viz3_correlation_heatmap.png")

def main():
    df = load_data()
    df_clean = clean_data(df)
    generate_visualizations(df_clean)
    print("\nPipeline execution complete.")

if __name__ == '__main__':
    main()
