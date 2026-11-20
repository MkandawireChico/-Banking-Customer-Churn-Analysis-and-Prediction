# Optional Power BI Dashboard Specification

## Status and scope

This document is a build specification for a separate Power BI implementation. The deployed Streamlit Command Centre is the primary interactive application in this repository.

The Power BI guide is intentionally technology-neutral where possible. Do not copy old metric values from this document into a report. Load current evaluation results from the metadata stored in `models/logistic_regression_pipeline.joblib`, or from a controlled exported metrics table.

## Recommended data sources

| Source | Use |
|---|---|
| `data/processed/churn_customers_clean.csv` | Customer attributes and actual churn outcome in the generated cohort |
| `data/processed/churn_customer_scores.csv` | Full-population portfolio scoring, risk distribution and prioritisation |
| `data/processed/churn_predictions.csv` | Independent held-out test predictions and actual outcomes |

Keep the evaluation and portfolio-scoring populations separate. Do not join the full-population scores to a model-performance visual and present the result as independent test performance.

## Suggested report pages

### 1. Executive portfolio overview
- Customer population and observed churn rate.
- Risk-band counts and shares.
- Churn exposure by customer segment and branch.
- High-risk customer count and balance exposure.
- Clear label identifying whether a visual uses the full portfolio or held-out test cohort.

### 2. Customer profile and behaviour
- Churn by age band, customer segment and branch.
- Churn by product count and tenure.
- Churn by digital engagement and transaction activity.
- Churn by complaint count and days since last transaction.

### 3. Retention prioritisation
- Ranked customer worklist based on full-population scores.
- Risk band, churn score, segment, branch and relevant behavioural context.
- Capacity selector or parameter for top-N outreach scenarios.
- Clear note that ranking is decision support, not an automatic customer action.

### 4. Model governance
- Held-out accuracy, precision, recall, F1 and balanced accuracy.
- ROC-AUC and average precision.
- Brier score, log loss, calibration curve and ECE.
- Confusion matrix and threshold scenarios.
- Subgroup diagnostics with minimum-group-size suppression.
- Validation-only threshold design clearly distinguished from final test evaluation.

### 5. Campaign and case operations
- Campaign volume and manager workload, if campaign exports are imported.
- Case status, overdue follow-ups and ownership, if case-register exports are imported.
- Treat imported CSVs as operational snapshots, not a live shared CRM.

## Suggested measures

Create measures against the relevant table rather than assuming a single fact table contains every population.

```dax
Total Customers = DISTINCTCOUNT(Customers[Customer_ID])

Observed Churned Customers =
CALCULATE(
    DISTINCTCOUNT(Customers[Customer_ID]),
    Customers[Churn] = 1
)

Observed Churn Rate =
DIVIDE([Observed Churned Customers], [Total Customers])

High Risk Customers =
CALCULATE(
    DISTINCTCOUNT(PortfolioScores[Customer_ID]),
    PortfolioScores[Risk_Band] = "High Risk"
)
```

For test metrics, use the held-out predictions table and a consistent threshold. For full-population monitoring, use the portfolio scores table and avoid describing its scores as independently validated performance.

## Data model guidance

A practical model can use:
- A customer dimension keyed by `Customer_ID`.
- A portfolio-score fact table keyed by `Customer_ID`.
- A held-out evaluation fact table keyed by `Customer_ID`.
- Optional branch and segment dimensions.
- Optional campaign and case tables populated from CSV exports.

Keep evaluation and portfolio fact tables distinct. Do not assume campaign assignments or case updates are stored by the Streamlit application.

## Governance notes

The underlying records and churn outcomes are synthetic. Findings are suitable for demonstrating analytics and dashboard design, not for claiming actual bank performance. Before operational use, validate on authorised representative data, assess fairness and calibration, monitor drift and require human review.
