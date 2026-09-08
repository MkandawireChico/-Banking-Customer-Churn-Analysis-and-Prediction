# Banking Customer Churn Analysis and Prediction
## Nyanja Commercial Bank (Fictional) — Malawi — Synthetic Portfolio Dataset

## Executive Summary

This project builds a complete customer-churn analytics workflow for a fictional mid-sized commercial bank in Malawi: synthetic data generation, cleaning, exploratory analysis, feature engineering, classification modeling (Logistic Regression and Random Forest), threshold analysis, model interpretation, individual customer risk scoring, and a retention action framework — ending in a Power BI dashboard specification for relationship managers and bank management.

## Business Problem

Customer churn creates financial and operational challenges for commercial banks because acquiring new customers typically costs more than retaining existing ones. The fictional bank wants to use historical customer information and behavioral indicators to identify customers at elevated churn risk and understand what's associated with attrition — combining exploratory analysis, statistical reasoning, machine learning, and business intelligence into a practical churn-risk solution.

## Malawi Banking Context

The synthetic data reflects a plausible Malawian retail/commercial banking mix: five customer segments (Retail, SME, Premium, Salary Account, Student/Youth) across ten branch locations. **Branch names (Blantyre, Lilongwe, Mzuzu, Zomba, Mangochi, Kasungu, Salima, Liwonde, Rumphi, Karonga) are real Malawian towns used as plausible location labels only** — no figures here represent actual banking activity in these locations.

## Project Objectives

1. Analyse customer demographics, product usage, transaction behavior, complaints, and account activity
2. Identify patterns and characteristics associated with churn
3. Clean and validate the customer dataset
4. Engineer meaningful behavioral features
5. Develop an interpretable Logistic Regression baseline and a Random Forest classifier
6. Compare model performance using appropriate evaluation metrics
7. Generate individual customer churn probabilities
8. Create meaningful, validated customer risk bands
9. Identify the most important predictors of churn
10. Translate findings into practical retention recommendations
11. Specify a Power BI dashboard for relationship managers and management

## Dataset

**20,018 customer records** (20,000 generated + ~1% deliberate duplicates, handled in cleaning) across 27 fields spanning demographics, products, digital banking, transaction behavior, complaints, loans, and the `Churn` target.

## Synthetic Data Disclaimer

**This project uses entirely synthetic data created for educational and portfolio purposes. The fictional bank, customers, figures, model results, and findings do not represent any real commercial bank or actual banking data in Malawi.** "Nyanja Commercial Bank" is the same fictional institution used in this portfolio's Loan Default Risk Prediction project, reused for narrative consistency — not because it represents a real bank.

## Data-Generation Methodology

**`Churn` is generated from a genuine logistic risk model, not random assignment.** The risk score combines recency of last transaction, transaction activity, complaint frequency, product count, account tenure, digital engagement, loan status, and customer segment — weighted and passed through a logistic function with random noise, so outcomes overlap realistically between churned and retained customers rather than being perfectly separable.

This was verified before any modeling began (notebook Section 1.4):

| Check | Result |
|---|---|
| Churn by days-since-last-transaction quartile | 7.5% → 9.2% → 11.9% → **21.7%** |
| Churn by number of complaints (0→4+) | 9.9% → 15.4% → 20.9% → 34.0% → **47.6%** |
| Churn by number of products (1→6) | **16.0%** → 10.4% → 8.3% → 5.0% → 4.4% → 1.7% |
| Churn by account tenure quartile | **17.8%** (newest) → 14.1% → 11.3% → 5.9% (longest) |
| Churn by mobile banking usage | 15.4% (no) vs. **11.2%** (yes) |

**Overall churn rate: 12.54%.**

## Data-Quality Issues (Deliberately Injected)

Eight categories, each reproducible via its own random seed: missing income (~2%), missing marital status (~1.5%), duplicate records (~1%), inconsistent segment labels/casing (~1.5%), implausible ages (values like 5, 130, -3; ~0.2%), extreme income outliers (10–20x; ~0.3%), invalid negative transaction counts (~0.3%), and missing internet-banking flags (~1.5%). None were removed silently.

## Data-Cleaning Process

- **Missing income, internet-banking flag:** left as missing in the cleaned CSV, imputed inside the modeling pipeline using training-set-only statistics — specifically to avoid leaking test-set information.
- **Missing marital status:** left missing, most-frequent imputed in the pipeline (a low-importance demographic field).
- **Duplicates:** exact duplicates dropped.
- **Inconsistent segment labels:** mapped to the 5 canonical categories via explicit lookup.
- **Implausible ages:** values outside 18–90 treated as missing (not guessed at), imputed downstream.
- **Extreme income:** winsorized at the 99.5th percentile.
- **Negative transaction counts:** a count cannot be negative by definition — treated as missing, not "corrected" by guessing the intended value.

## Preventing Data Leakage

**This is addressed explicitly, not as a footnote.** The fields that would constitute leakage — `Account_Closed`, `Exit_Interview_Result`, `Post_Churn_Balance`, `Closure_Date`, `Retention_Outcome` — were **deliberately never generated** in this dataset. These are exactly the kind of fields that only exist *because* a customer already churned; a model trained on them would trivially "predict" churn using information that's a consequence of the outcome, not a precursor of it. Everything actually in this dataset (transaction activity, complaints, product count, tenure, digital engagement) is a pre-churn behavioral signal the bank would genuinely have *before* knowing the outcome — which is the entire point of a predictive, not descriptive, churn model.

## Exploratory Data Analysis

16 cuts of the data (age, gender, segment, branch, tenure, products, mobile/internet banking, transaction activity, complaints, balance, income, loan status, digital engagement) — 7 exported charts in `visuals/charts/`. Full detail in the notebook, Section 7.

## Feature Engineering

| Feature | Definition | Banking meaning |
|---|---|---|
| `Average_Transaction_Value` | Total 6-month value ÷ (monthly transactions × 6) | typical transaction size |
| `Transaction_Activity_Index` | Monthly transactions ÷ (days since last transaction + 1) | combined activity *and* recency in one number |
| `Digital_Banking_Usage` | Mobile + internet banking flags (0/1/2) | breadth of digital channel adoption |
| `Complaint_Rate` | Complaints ÷ years of tenure | complaints per year of relationship, not raw count |
| `Engagement_Index` | Average of scaled digital score and normalized activity | single composite engagement measure |
| `Balance_to_Income_Ratio` | Balance ÷ income | how much of a customer's financial life sits with this bank |
| `Products_per_Tenure` | Products ÷ years of tenure | adoption rate, not just raw count |
| `Recent_Activity_Flag` | 1 if inactive ≤30 days | simple, actionable recency cutoff |
| `High_Value_Customer_Flag` | 1 if balance in top quartile | flags customers whose departure matters most financially |
| `Low_Engagement_Flag` | 1 if digital score < 40 | simple, actionable digital-risk flag |

## Modelling Methodology

- **Stratified 75/25 train/test split** — preserves the 12.5% churn rate in both sets.
- **Class imbalance:** addressed via `class_weight="balanced"` and stratification. **Oversampling (e.g. SMOTE) was deliberately not used** — the brief specifically asked not to apply it without demonstrated need, and `class_weight` plus stratification proved sufficient to get meaningful minority-class recall (Section 9).
- **Leakage-aware pipeline:** median/most-frequent imputation (numeric/categorical) fit on the training set only, then scaling/one-hot encoding.
- **Excluded from modeling:** `Customer_ID` (no predictive meaning), `Occupation` (identical to `Employment_Status` in this dataset), and the five leakage fields discussed above (which don't exist in the dataset at all).

## Model Evaluation

**Why accuracy alone is misleading:** with a 12.5% churn rate, a trivial "predict retained for everyone" model scores ~87.5% accuracy while catching zero churners — useless for a retention program. Precision, recall, F1, and ROC-AUC on the Churn class are evaluated together instead.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.677 | 0.228 | **0.661** | 0.339 | **0.740** |
| Random Forest | 0.741 | 0.260 | 0.576 | 0.359 | 0.735 |

**Selected model: Logistic Regression** — chosen for its higher recall and marginally higher ROC-AUC, and because for a retention program a missed high-risk customer (false negative) is typically more costly than an unnecessary outreach (false positive). This is a business-objective decision, not an automatic "highest accuracy wins" — Random Forest actually has higher raw accuracy (0.741 vs. 0.677) but at the cost of catching fewer real churners, which matters more for this use case.

## Threshold Analysis

The default 0.50 threshold was not assumed optimal. Five thresholds were compared on the selected model:

| Threshold | Precision | Recall | F1 | False Positives | False Negatives | Customers Flagged |
|---|---|---|---|---|---|---|
| 0.30 | 0.158 | **0.924** | 0.270 | 3,095 | 48 | 3,675 |
| 0.40 | 0.187 | 0.815 | 0.304 | 2,231 | 116 | 2,743 |
| 0.50 (default) | 0.228 | 0.661 | 0.339 | 1,405 | 213 | 1,820 |
| 0.60 | 0.288 | 0.495 | 0.364 | 768 | 317 | 1,079 |
| 0.70 | 0.364 | 0.325 | 0.343 | 356 | 424 | 560 |

**Business trade-off:** lowering the threshold catches dramatically more real churners (recall jumps from 66.1% to 92.4% moving from 0.50 to 0.30) at the cost of many more false positives to review. Given that a missed churner is a lost relationship with zero intervention attempted — while a false positive only costs a relationship manager's outreach time — **a threshold in the 0.30–0.40 range is more defensible than the default 0.50** for this specific use case. The exact choice ultimately depends on the bank's real outreach capacity (how many customers relationship managers can contact per month), which this synthetic project has no way to know.

## Model Interpretation

**Logistic Regression** (top risk-increasing factors): `Days_Since_Last_Transaction`, `Number_of_Complaints`, `Average_Monthly_Deposit`, being in the `Student/Youth` segment. Top risk-*decreasing* factors: tenure, product count, and mobile/internet banking usage.

**Random Forest** independently ranks `Days_Since_Last_Transaction` as its single most important feature too — agreement between two structurally different models is meaningful evidence this is a genuine signal, not an artifact of one algorithm.

**An honest interpretability caveat, not glossed over:** `Digital_Engagement_Score` shows a small *positive* (risk-increasing) coefficient in Logistic Regression — the opposite of what the underlying data-generating process actually built in (digital engagement was designed to *reduce* churn risk). This is very likely a **multicollinearity artifact**: `Digital_Engagement_Score` is highly correlated with `Has_Mobile_Banking` and `Has_Internet_Banking`, which are *also* in the model, so linear regression can split and even invert a shared effect's sign across correlated features. Random Forest — which doesn't have this sign-instability problem, since importance is always non-negative — ranks `Digital_Engagement_Score` with modest, unambiguous importance instead. This is exactly the kind of thing worth catching and reporting honestly rather than presenting the raw coefficient table without comment.

**Association vs. prediction vs. causation, maintained explicitly throughout:** the EDA shows association (patterns in who has historically churned); the models produce a prediction (an estimated probability for a given customer); neither establishes causation. "More complaints correlates with more churn" does not mean "resolving complaints will proportionally reduce churn" — that would require a controlled test (e.g. an A/B outreach experiment), which this project does not run.

## Key Findings

1. **Recency of activity is the strongest single predictor**, confirmed independently by both models: churn rises from 7.5% to 21.7% across days-since-last-transaction quartiles. *(Prediction-relevant, and directly actionable — recency is observable in real time.)*
2. **Complaints show the steepest gradient of any factor examined**: churn rises from 9.9% (no complaints) to 47.6% (4+ complaints) — a nearly 5x spread. *(Association — the EDA doesn't establish that resolving complaints would proportionally cut churn, only that they co-occur strongly.)*
3. **Product count is protective, and the effect is large**: single-product customers churn at 16.0% vs. 1.7% for customers with 6 products. *(Association — deeper relationships correlate with retention, though customers who already trust the bank more may simply be more likely to both hold more products and stay, regardless of the products themselves.)*
4. **High-value customers are not immune — and also not disproportionately at risk.** Top-balance-quartile customers churn at 12.53%, essentially identical to the 12.55% overall rate (SQL query `sql/analysis_queries.sql` Q8b). This directly satisfies the brief's expectation that "higher-value customers should not automatically have zero churn risk" — but the more precise, more useful finding is that account value and churn risk are close to *independent* in this data, which argues for allocating retention effort by *risk band*, not by customer value.
5. **A genuine interpretability caveat was found and reported, not hidden**: Logistic Regression's coefficient for `Digital_Engagement_Score` has the "wrong" sign due to multicollinearity with two correlated digital-adoption flags also in the model — resolved by cross-checking against Random Forest's importance ranking, which doesn't share that instability. *(A methodology finding, not a churn-driver finding — but an important one for anyone reading the coefficient table at face value.)*
6. **Model selection was a genuine trade-off, not a clean win for one model**: Random Forest has higher accuracy (0.741 vs. 0.677) but Logistic Regression has meaningfully higher recall (0.661 vs. 0.576) and a marginally better ROC-AUC — Logistic Regression was selected for the business reason stated above, not because it dominated on every metric.
7. **The validated risk bands separate cleanly**: Low/Medium/High risk (by predicted probability) show actual test-set churn rates of **3.6%, 10.4%, and 28.8%** respectively — confirming the bands mean what they claim before they're used for anything.

## Business Recommendations

- **Prioritize outreach by recency and complaint history first** — the two strongest, most actionable signals found (Findings #1–2).
- **Treat the 0.30–0.40 threshold range as the operating point for flagging customers**, not the default 0.50, given the retention-specific cost asymmetry (missed churner > wasted outreach) — but validate actual outreach capacity before committing to a specific cutoff.
- **Don't assume high-value customers are safe, and don't over-target them either** — Finding #4 shows value and risk are largely independent, so a value-weighted retention budget would misallocate effort.
- **Cross-check any single model's coefficients before acting on them** — Finding #5 is a concrete example of how a plausible-looking coefficient can be a statistical artifact rather than a real business signal.
- **Treat all findings as association or prediction, never as proof of what would work** — any specific retention intervention (a call campaign, a fee waiver, a product bundle) should be tested (e.g. via a pilot with a control group) before being scaled, since nothing in this analysis establishes causation.

## Retention Action Framework

**These are analytical recommendations only — not actual policies of any real bank.**

| Risk + Behavioral Profile | Recommended Action |
|---|---|
| High Risk + Low Transaction Activity | Relationship manager outreach — a personal check-in |
| High Risk + High Complaints | Customer service intervention — resolve the complaint before any retention messaging |
| High Risk + Low Product Adoption | Product education / relevant product recommendation |
| Medium Risk + Declining Digital Engagement | Lighter-touch digital engagement campaign |
| Low Risk + High Engagement | Maintain relationship and monitor — no action needed |

## Power BI Dashboard

A 4-page specification (`dashboard/powerbi_dashboard_guide.md`, not yet built): Executive Overview, Customer Profile, Churn Drivers, and High-Risk Customers — designed around the specific questions a relationship manager needs answered ("who do I contact first, and why"), including a planned callout for Finding #4 (value ≠ risk) directly on the dashboard.

## Limitations

- This is a cross-sectional synthetic snapshot, not a real historical customer base — there is no genuine time dimension, so a "churn trend" visual isn't available without extending the simulation.
- The specific figures throughout this README (12.54% churn rate, 0.740 ROC-AUC, etc.) are properties of this simulation's parameters and random seed, not measurements of any real bank's customers.
- Real customer sentiment (satisfaction surveys, actual complaint text, competitor offers) isn't modeled — only structured behavioral proxies are.
- The Logistic Regression multicollinearity issue (Finding #5) is a reminder that this model's individual coefficients should be read cautiously for correlated features, even though its overall predictive performance is sound.
- False positives and false negatives both carry real costs this project can only reason about qualitatively — actual deployment needs the bank's real cost estimates.
- Prediction does not establish causation (repeated deliberately throughout this README) — retention actions based on these findings should be piloted and measured, not assumed to work.
- Model performance on real banking data would likely differ from these synthetic results in ways that can't be anticipated from synthetic data alone.

## Ethical Considerations

**This project uses entirely synthetic data created for educational and portfolio purposes.** No real customer data was used at any point. In a real deployment, churn models raise genuine ethical considerations this project does not claim to have resolved: customer privacy and data protection, avoiding discriminatory outcomes (e.g. a model that inadvertently penalizes customers by proxy for protected characteristics), mandatory human review before any customer-facing action is taken on a high-risk flag, and ongoing model monitoring and revalidation as customer behavior and the bank's own products change. **This project does not recommend automatically closing accounts, denying services, or taking punitive action based solely on a churn prediction** — every recommendation here assumes a human relationship manager or service team remains in the loop.

## Technologies Used

Python (Pandas, NumPy, Matplotlib, Seaborn, SciPy, Scikit-learn, joblib) · SQL (SQLite) · Jupyter Notebook · Power BI (specification)

## Repository Structure

```
banking-customer-churn-analysis/
├── data/
│   ├── raw/
│   │   └── churn_customers_synthetic.csv       # As generated + injected issues
│   └── processed/
│       ├── churn_customers_clean.csv            # Cleaned + feature-engineered
│       └── churn_predictions.csv                # Customer_ID, Actual_Churn, Churn_Probability, Predicted_Churn, Risk_Band
├── notebooks/
│   └── customer_churn_analysis.ipynb            # Full workflow, executed with real outputs
├── scripts/
│   └── generate_dataset.py                       # Standalone, reproducible data generator
├── models/
│   ├── logistic_regression_pipeline.joblib
│   └── random_forest_pipeline.joblib
├── sql/
│   ├── analysis_queries.sql                      # 13 queries, CTEs + window functions
│   ├── churn_analysis.db
│   └── README.md
├── dashboard/
│   └── powerbi_dashboard_guide.md                # Specification (not yet built)
├── visuals/
│   └── charts/                                    # 11 chart exports from the notebook
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation Instructions

```bash
git clone <this-repo>
cd banking-customer-churn-analysis
pip install -r requirements.txt
```

## How to Reproduce the Project

```bash
# Option A: run the full notebook (generates data + trains models + exports everything)
jupyter notebook notebooks/customer_churn_analysis.ipynb

# Option B: regenerate just the raw dataset independently
cd scripts
python generate_dataset.py

# SQL analysis
cd sql
sqlite3 churn_analysis.db < analysis_queries.sql
```

Both data-generation paths use `np.random.seed(42)` and were verified to reproduce the identical 20,200-row raw dataset (12.54% pre-cleaning churn rate) described in this README.

## Author

Chico Mkandawire
MSc Data Science, Malawi University of Science and Technology (in progress)
BSc Management Information Systems, Malawi University of Business and Applied Sciences
