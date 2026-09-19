# Telco Customer Retention Analytics & Visual Storytelling

This repository contains the end-to-end data preparation, exploratory data analysis, and advanced multi-variable data storytelling pipeline for the **Telco Customer Churn Dataset** (7,043 subscriber records across 21 attributes).

---

## Repository Structure

```text
├── analysis.py                         # Week 1: Data Acquisition & Baseline Cleaning
├── week2_analysis.py                   # Week 2: Multi-Variable Visualizations Pipeline
├── requirements.txt                    # Project Dependencies
├── viz1_compound_contract_internet_risk.png # Compound Attrition Risk Bar Chart
├── viz2_monthly_charges_payment_violin.png  # Price Elasticity Across Payment Channels
├── viz3_survival_trajectory_tenure.png      # Stepped Retention Survival Trajectory
├── viz4_service_bundling_heatmap.png        # Value-Add Ecosystem Protection Matrix
├── viz5_tenure_vs_monthly_scatter.png       # High-Value Risk Matrix Scatter Plot
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

Execution Instructions
Bash
# Install dependencies
pip install -r requirements.txt

# Run Week 1 Ingestion & Cleaning Script
python analysis.py

# Run Week 2 Storytelling Pipeline
python week2_analysis.py
