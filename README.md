# Week 1: Data Acquisition, Cleaning, and Exploratory Data Analysis (EDA)

## Project Overview
This repository contains the end-to-end data preparation and exploratory analysis pipeline for customer retention analytics using the Telco Customer Churn dataset (7,043 subscriber records across 21 features).

## Repository Structure
├── analysis.py                          # Full data loading, cleaning, and plotting pipeline
├── requirements.txt                     # Project dependencies
├── viz1_contract_churn_rate.png         # Contract horizon vs. churn bar plot
├── viz2_tenure_density_distribution.png # Tenure survival density KDE plot
├── viz3_correlation_heatmap.png         # Correlation matrix of numerical features
└── README.md                            # Documentation and findings
## Data Cleaning Methodology
* **Type Parsing & Latent Null Resolution:** The `TotalCharges` field contained 11 whitespace strings (`" "`) belonging to brand-new accounts (`tenure = 0`). These strings caused the numeric feature to parse as an object type. Values were coerced to numeric and imputed with `0.0` to preserve customer records without distorting distributions.
* **Integrity Audit:** Verified zero duplicate records across the unique `customerID` index.
* **Target Encoding:** Mapped `Churn` ('Yes'/'No') to binary indicators (`1`/`0`) for numeric correlation analysis.

## Key Insights
1. **Contractual Vulnerability:** Month-to-month contracts experience an attrition rate of 42.71%, compared to 2.83% for two-year contract holders.
2. **Early Lifecycle Churn Spike:** Kernel density estimation confirms that over 50% of customer loss occurs within the initial 9 to 12 months of service.
3. **Price Sensitivity:** Monthly charges show a positive correlation with churn ($r = +0.19$), whereas tenure demonstrates a strong protective relationship ($r = -0.35$).
