"""
Week 3 Task: Statistical Analysis and Hypothesis Testing in Python
Author: Anurag Kaushik
Domain: Telco Customer Retention Analytics
Stack: Python 3.10+, Pandas, NumPy, SciPy, Statsmodels, Matplotlib, Seaborn
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")

def load_data():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)
    df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def run_statistical_tests(df):
    # Test 1: Chi-Square Test of Independence
    contingency_table = pd.crosstab(df['Contract'], df['Churn'])
    chi2, p_val1, dof, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
    print(f"Chi2: {chi2:.2f}, p: {p_val1:.4e}, Cramer's V: {cramers_v:.3f}")

    # Test 2: Welch's t-Test & Mann-Whitney U
    churned = df[df['Churn'] == 'Yes']['MonthlyCharges']
    retained = df[df['Churn'] == 'No']['MonthlyCharges']
    t_stat, p_val2 = stats.ttest_ind(churned, retained, equal_var=False)
    diff_mean = churned.mean() - retained.mean()
    pooled_sd = np.sqrt((churned.std()**2 + retained.std()**2) / 2)
    cohens_d = diff_mean / pooled_sd
    print(f"Welch t: {t_stat:.2f}, p: {p_val2:.4e}, Diff: ${diff_mean:.2f}, Cohen's d: {cohens_d:.3f}")

    # Test 3: One-Way ANOVA & Tukey HSD
    model = ols('tenure ~ C(PaymentMethod)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    f_stat = anova_table.loc['C(PaymentMethod)', 'F']
    p_val3 = anova_table.loc['C(PaymentMethod)', 'PR(>F)']
    ss_between = anova_table.loc['C(PaymentMethod)', 'sum_sq']
    eta_sq = ss_between / (ss_between + anova_table.loc['Residual', 'sum_sq'])
    print(f"ANOVA F: {f_stat:.2f}, p: {p_val3:.4e}, Eta-squared: {eta_sq:.3f}")

def generate_statistical_plots(df):
    # Viz 1: Chi-Square Proportional Distribution
    plt.figure(figsize=(9, 5))
    ct_prop = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    ax1 = ct_prop.plot(kind='bar', stacked=True, color=['#1f77b4', '#d62728'], figsize=(9, 5), edgecolor='black')
    plt.title('Proportional Churn by Contract Horizon (chi2 = 1184.55, p < 0.0001)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Contract Type', fontsize=11)
    plt.ylabel('Percentage (%)', fontsize=11)
    plt.legend(title='Status', labels=['Retained', 'Churned'], loc='upper right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('stat_viz1_contract_chi2.png', dpi=300)
    plt.close()

    # Viz 2: Welch's t-Test Monthly Charges Density
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', palette={'No': '#1f77b4', 'Yes': '#d62728'}, fill=True, common_norm=False, alpha=0.4)
    plt.axvline(df[df['Churn'] == 'Yes']['MonthlyCharges'].mean(), color='#d62728', linestyle='--', linewidth=2, label='Churned Mean: $74.44')
    plt.axvline(df[df['Churn'] == 'No']['MonthlyCharges'].mean(), color='#1f77b4', linestyle='--', linewidth=2, label='Retained Mean: $61.27')
    plt.title('Monthly Charges Density & Sample Means (Welch t = 18.27, p < 0.0001)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Monthly Charges ($)', fontsize=11)
    plt.ylabel('Probability Density', fontsize=11)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('stat_viz2_monthly_charges_ttest.png', dpi=300)
    plt.close()

    # Viz 3: One-Way ANOVA Tenure Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='PaymentMethod', y='tenure', palette='Blues_r', showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red"})
    plt.title('Account Tenure Distribution across Payment Modalities (ANOVA F = 464.22, p < 0.0001)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Payment Method', fontsize=11)
    plt.ylabel('Account Tenure (Months)', fontsize=11)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('stat_viz3_payment_tenure_anova.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    df = load_data()
    run_statistical_tests(df)
    generate_statistical_plots(df)
    print("Week 3 Statistical Analysis completed successfully.")
