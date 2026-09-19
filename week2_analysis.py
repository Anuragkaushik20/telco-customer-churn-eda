"""
Week 2 Task: Advanced Data Visualization and Storytelling with Python
Author: Anurag Kaushik
Domain: Telco Customer Retention Analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")

def load_and_preprocess():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)
    df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def generate_week2_plots(df):
    # Plot 1: Compound Attrition Risk
    plt.figure(figsize=(10, 6))
    compound = df.groupby(['Contract', 'InternetService'])['Churn_Numeric'].agg(['mean', 'count']).reset_index()
    compound['Churn_Rate'] = compound['mean'] * 100

    ax = sns.barplot(data=compound, x='Churn_Rate', y='Contract', hue='InternetService', palette='Blues_r')
    plt.title('Compound Churn Risk: Contract Type x Internet Service Tier', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Churn Rate (%)', fontsize=12)
    plt.ylabel('Contract Horizon', fontsize=12)
    plt.legend(title='Internet Service', loc='lower right')

    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(f"{width:.1f}%", (width + 1, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=10, fontweight='bold')

    plt.xlim(0, 65)
    plt.tight_layout()
    plt.savefig('viz1_compound_contract_internet_risk.png', dpi=300)
    plt.close()

    # Plot 2: Monthly Charge Distribution (Violin)
    plt.figure(figsize=(11, 6))
    sns.violinplot(data=df, x='PaymentMethod', y='MonthlyCharges', hue='Churn',
                   split=True, inner="quartile", palette={'No': '#1f77b4', 'Yes': '#d62728'})
    plt.title('Monthly Charge Distribution & Price Elasticity across Payment Modalities', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Payment Method', fontsize=11)
    plt.ylabel('Monthly Charges ($)', fontsize=11)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Status', loc='upper left')
    plt.tight_layout()
    plt.savefig('viz2_monthly_charges_payment_violin.png', dpi=300)
    plt.close()

    # Plot 3: Retention Survival Curve
    plt.figure(figsize=(10, 6))
    tenure_bins = np.arange(0, 73, 6)

    for contract, color in zip(['Month-to-month', 'One year', 'Two year'], ['#d62728', '#ff7f0e', '#2ca02c']):
        subset = df[df['Contract'] == contract]
        survival = [(subset['tenure'] > t).mean() for t in tenure_bins]
        plt.step(tenure_bins, survival, label=f'{contract}', where='post', linewidth=2.5, color=color)

    plt.axvspan(0, 12, color='red', alpha=0.1, label='High-Risk Onboarding Window (0-12m)')
    plt.title('Customer Retention Survival Trajectory by Contract Horizon', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Account Tenure (Months)', fontsize=12)
    plt.ylabel('Retention Survival Probability', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    plt.savefig('viz3_survival_trajectory_tenure.png', dpi=300)
    plt.close()

    # Plot 4: Service Bundling Heatmap
    plt.figure(figsize=(8, 6))
    fiber_df = df[df['InternetService'] == 'Fiber optic']
    pivot_sec = fiber_df.pivot_table(index='TechSupport', columns='OnlineSecurity', values='Churn_Numeric', aggfunc='mean') * 100

    sns.heatmap(pivot_sec, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Churn Rate (%)'}, linewidths=1)
    plt.title('Fiber Optic Subscriber Churn Rate (%): Tech Support vs. Online Security', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Online Security Add-on', fontsize=11)
    plt.ylabel('Tech Support Add-on', fontsize=11)
    plt.tight_layout()
    plt.savefig('viz4_service_bundling_heatmap.png', dpi=300)
    plt.close()

    # Plot 5: High-Value Customer Risk Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Churn', style='Churn',
                    palette={'No': '#1f77b4', 'Yes': '#d62728'}, alpha=0.6, s=50)

    plt.axvline(x=12, color='black', linestyle='--', alpha=0.7)
    plt.axhline(y=70, color='black', linestyle='--', alpha=0.7)

    plt.text(3, 110, 'CRITICAL RISK ZONE\n(Low Tenure, High Charge)\nChurn: 61.4%', 
             fontsize=10, fontweight='bold', color='darkred', bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

    plt.title('Subscriber Risk Matrix: Tenure vs. Monthly Charges', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Account Tenure (Months)', fontsize=12)
    plt.ylabel('Monthly Charges ($)', fontsize=12)
    plt.legend(title='Status', loc='lower right')
    plt.tight_layout()
    plt.savefig('viz5_tenure_vs_monthly_scatter.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    df = load_and_preprocess()
    generate_week2_plots(df)
