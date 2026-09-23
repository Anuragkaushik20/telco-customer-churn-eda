# Telco Customer Retention Analytics, Storytelling & Hypothesis Testing

This repository contains the end-to-end data processing, exploratory analysis, multi-variable visual storytelling, and inferential statistical testing pipeline for the **Telco Customer Churn Dataset** (7,043 subscriber records across 21 attributes).

---

## Repository Structure

```text
├── analysis.py                         # Week 1: Ingestion & Baseline Cleaning
├── week2_analysis.py                   # Week 2: Advanced Visualizations Pipeline
├── week3_analysis.py                   # Week 3: Inferential Hypothesis Testing Pipeline
├── requirements.txt                    # Project Dependencies
├── viz1_compound_contract_internet_risk.png # Week 2 Plot 1
├── viz2_monthly_charges_payment_violin.png  # Week 2 Plot 2
├── viz3_survival_trajectory_tenure.png      # Week 2 Plot 3
├── viz4_service_bundling_heatmap.png        # Week 2 Plot 4
├── viz5_tenure_vs_monthly_scatter.png       # Week 2 Plot 5
├── stat_viz1_contract_chi2.png         # Week 3 Plot 1 (Chi-Square)
├── stat_viz2_monthly_charges_ttest.png # Week 3 Plot 2 (Welch's t-Test)
├── stat_viz3_payment_tenure_anova.png  # Week 3 Plot 3 (ANOVA)
└── README.md                           # Master Project Documentation



Week 1 Overview: Data Integrity & Cleaning Audit
Type Enforcement & Latent Null Resolution: Isolated 11 whitespace string records (" ") in TotalCharges corresponding to zero-tenure accounts (tenure = 0). Values were coerced to float64 and imputed with 0.0 to preserve account records without skewing distributions.

Integrity Audit: Verified zero primary key collisions across 7,043 unique customerID records.

Target Mapping: Converted binary string target Churn ('Yes'/'No') to integer indicators (1/0) for numeric correlation analysis.

Week 2 Deliverables: Multi-Variable Visual Storytelling
1. Compound Attrition Risk (Contract Horizon × Internet Technology)
Finding: Fiber Optic subscribers on Month-to-Month contracts suffer an extreme 54.6% churn rate (1,162 / 2,128 accounts), accounting for 62.0% of all churned accounts in the enterprise. Two-Year contract holders on Fiber Optic drop to a 5.8% churn rate.

2. Price Elasticity & Payment Method Volatility
Finding: The median monthly charge for churned customers using Electronic Check is $86.50, compared to $46.30 for retained bank transfer accounts. Electronic check users show high sensitivity to unbundled premium charges.

3. Customer Retention Survival Trajectory
Finding: Attrition risk is heavily concentrated within the 0 to 12-month tenure window for month-to-month contracts, where survival probability drops steeply to 53.1%.

4. Ecosystem Protection Matrix (Service Bundling)
Finding: Fiber Optic subscribers with neither Tech Support nor Online Security churn at 49.3%. Bundling both add-ons reduces churn to 18.7%, proving ecosystem lock-in mitigates price sensitivity.

5. Subscriber Risk Quadrant Analysis
Finding: The High Charge (> $70/mo) & Low Tenure (< 12 months) quadrant holds 1,024 accounts with an average churn rate of 61.4%, representing $860,000 in annualized revenue at risk.

Multi-Variable Risk Summary Table
Customer Segment	Total Accounts	Churned Accounts	Churn Rate (%)	Revenue Impact	Primary Risk Driver
Fiber Optic + Month-to-Month	2,128	1,162	54.6%	Critical ($103k/mo)	Lack of contract lock-in + unbundled high fees
Electronic Check Payment	2,365	1,071	45.3%	High ($92k/mo)	Friction-heavy payment channel, unautomated billing
Fiber Optic w/ No Tech Support	3,387	1,407	41.5%	High ($119k/mo)	Lack of product onboarding & technical assistance
Two-Year Contract Holders	1,695	48	2.8%	Low (Stable)	Strong contract lock-in & loyalty incentives
Strategic Action Plan
Automated Onboarding Protocols: Deploy automated 90-day onboarding workflows offering 3-month free trials of TechSupport to new Fiber Optic subscribers.

Auto-Pay Migration Incentive: Offer a $5/month discount to shift Electronic Check users onto automated credit card / ACH payments.

Month-6 Contract Conversion: Trigger automated marketing incentives at Month 6 of tenure to convert Month-to-Month users into 12-month contracts.

Potential Areas for Further Investigation & Next Steps
Survival Modeling (Cox Proportional Hazards): Fit a Cox Proportional Hazards model to estimate exact Hazard Ratios (HR) for feature risk factors over time.

Predictive Classification Pipelines: Train XGBoost, LightGBM, and Random Forest models applying SMOTE (Synthetic Minority Over-sampling Technique) to handle target class imbalance (73.5% / 26.5%).

Customer Lifetime Value (CLV) Segmentation: Perform K-Means clustering on behavioral and spend metrics to build a dollar-quantified CLV risk scoring matrix.


Week 3 Deliverables: Inferential Statistical Hypothesis Testing
1. Chi-Square ($\chi^2$) Test of Independence (Contract Horizon vs. Churn)
  Formal Hypotheses: $H_0$: Contract type and Churn status are independent vs. $H_1$: Significant structural association.
  Statistical Metrics: $\chi^2 = 1184.55$, $df = 2$, $p = 7.32 \times 10^{-258}$ ($p < 0.0001$).
  Effect Size: Cramér's $V = 0.410$ (Large Effect Size).
  Finding: Reject $H_0$. Month-to-month contract holders suffer a 42.71% churn rate, compared to 2.83% for Two-Year contract holders.
2. Welch's Two-Sample $t$-Test & Mann-Whitney U Test (Monthly Charges by Churn)
  Formal Hypotheses: $H_0: \mu_{\text{churned}} \le \mu_{\text{retained}}$ vs. $H_1: \mu_{\text{churned}} > \mu_{\text{retained}}$.
  Statistical Metrics: Welch $t = 18.27$, $df = 4212.8$, $p = 2.74 \times 10^{-72}$ ($p < 0.0001$).Mann-Whitney $U = 3,456,120.0$, $p < 0.0001$.
  Effect Size & CI: Cohen's $d = 0.445$. Mean difference = $+\$13.17/\text{month}$ ($95\%\text{ CI}: [\$11.76, \$14.58]$).
  Finding: Reject $H_0$. Churned customers pay significantly higher average monthly charges ($\$74.44$) than retained customers ($\$61.27$).
3. One-Way ANOVA & Post-Hoc Tukey HSD (Tenure across Payment Channels)
  Formal Hypotheses: $H_0: \mu_1 = \mu_2 = \mu_3 = \mu_4$ vs. $H_1$: At least one group mean differs.
  Statistical Metrics: $F(3, 7039) = 464.22$, $p = 3.22 \times 10^{-267}$ ($p < 0.0001$). Eta-squared $\eta^2 = 0.165$.
  Tukey HSD Key Result: Automated payment methods (Bank transfer and Credit card) yield over 18 months higher mean tenure than manual Electronic check payments ($p < 0.0001$).
  Finding: Reject $H_0$. Payment channel choice accounts for $16.5\%$ of total subscriber tenure variance.


Execution Instructions
Bash
# Install dependencies
pip install -r requirements.txt

# Run Week 1 Ingestion & Cleaning Script
python analysis.py

# Run Week 2 Storytelling Pipeline
python week2_analysis.py
