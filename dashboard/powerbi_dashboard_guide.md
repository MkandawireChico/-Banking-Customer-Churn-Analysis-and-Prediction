# Power BI Dashboard Specification

## Overview
This document specifies the Power BI dashboard for the Banking Customer Churn Analysis project. The dashboard is designed to provide business stakeholders with actionable insights into customer churn risk and retention opportunities.

**Note:** This is a specification document. The actual Power BI dashboard is not implemented in this repository. This specification can be used as a guide for building the dashboard in Power BI Desktop.

## Data Sources

### Primary Data Source
- **File:** `data/processed/churn_predictions.csv`
- **Description:** Customer-level predictions with churn probabilities, risk bands, and actual outcomes
- **Key Columns:**
  - `Customer_ID`: Unique customer identifier
  - `Actual_Churn`: Actual churn status (0 = retained, 1 = churned)
  - `Churn_Probability`: Model-predicted churn probability
  - `Predicted_Churn_0.50`: Binary prediction at 0.50 threshold
  - `Risk_Band`: Risk category (Low Risk, Medium Risk, High Risk)
  - All original customer features (demographics, account details, transaction behavior, etc.)

### Secondary Data Source
- **File:** `data/processed/churn_customers_clean.csv`
- **Description:** Full cleaned dataset with all features
- **Use Case:** Detailed customer profile analysis and drill-through

### Model Metadata
- **File:** `models/logistic_regression_pipeline.joblib` (contains metadata)
- **Description:** Model performance metrics, threshold analysis, capacity analysis
- **Use Case:** Display model performance metrics on dashboard

## Data Model

### Star Schema
```
FactCustomer (fact table)
├── Customer_ID (key)
├── Actual_Churn
├── Churn_Probability
├── Predicted_Churn_0.50
├── Risk_Band
├── All numeric features
└── Foreign keys to dimension tables

DimBranch (dimension)
├── Branch_Location
├── Branch_Region (derived)
└── Branch_Size (derived)

DimSegment (dimension)
├── Customer_Segment
└── Segment_Category (derived)

DimTime (dimension - optional for time-based analysis)
├── Tenure_Bucket
└── Activity_Bucket
```

## Key Measures (DAX)

### Churn Metrics
```dax
Churn Rate = DIVIDE([Churned Customers], [Total Customers])
Churned Customers = COUNTROWS(FILTER(FactCustomer, FactCustomer[Actual_Churn] = 1))
Total Customers = COUNTROWS(FactCustomer)
```

### Risk Band Metrics
```dax
High Risk Customers = COUNTROWS(FILTER(FactCustomer, FactCustomer[Risk_Band] = "High Risk"))
High Risk Churn Rate = DIVIDE(
    CALCULATE([Churned Customers], FactCustomer[Risk_Band] = "High Risk"),
    CALCULATE([Total Customers], FactCustomer[Risk_Band] = "High Risk")
)
Medium Risk Customers = COUNTROWS(FILTER(FactCustomer, FactCustomer[Risk_Band] = "Medium Risk"))
Low Risk Customers = COUNTROWS(FILTER(FactCustomer, FactCustomer[Risk_Band] = "Low Risk"))
```

### Model Performance Metrics
```dax
Model Accuracy = DIVIDE(
    COUNTROWS(FILTER(FactCustomer, FactCustomer[Predicted_Churn_0.50] = FactCustomer[Actual_Churn])),
    [Total Customers]
)
Model Precision = DIVIDE(
    CALCULATE([Churned Customers], FactCustomer[Predicted_Churn_0.50] = 1),
    CALCULATE([Total Customers], FactCustomer[Predicted_Churn_0.50] = 1)
)
Model Recall = DIVIDE(
    CALCULATE([Churned Customers], FactCustomer[Predicted_Churn_0.50] = 1),
    [Churned Customers]
)
```

### Capacity-Constrained Metrics
```dax
// For capacity of 500 customers (top 500 by churn probability)
Top 500 Precision = VAR Top500 = 
    TOPN(500, FactCustomer, FactCustomer[Churn_Probability], DESC)
RETURN
    DIVIDE(
        COUNTROWS(FILTER(Top500, FactCustomer[Actual_Churn] = 1)),
        COUNTROWS(Top500)
    )
```

## Dashboard Pages

### Page 1: Executive Overview
**Purpose:** High-level churn metrics and trends

**Visuals:**
1. **KPI Cards:**
   - Total Customers
   - Overall Churn Rate
   - High Risk Customers (count and %)
   - Model ROC-AUC Score (static value from model metadata: 0.74)

2. **Churn Rate by Segment:** Bar chart showing churn rate across customer segments

3. **Churn Rate by Branch:** Map or bar chart showing geographic distribution

4. **Risk Band Distribution:** Donut chart showing Low/Medium/High risk distribution

5. **Churn Trend by Tenure:** Line chart showing churn rate by account tenure buckets

6. **Model Performance Comparison:** Table comparing baseline vs model metrics:
   - Baseline Accuracy: 87.5% (always predicts retained)
   - Model Accuracy: 67.7%
   - Baseline Recall: 0% (never catches churn)
   - Model Recall: 65.3%
   - Model ROC-AUC: 0.74

### Page 2: Customer Profile Analysis
**Purpose:** Understand churn drivers by customer characteristics

**Visuals:**
1. **Churn by Age Group:** Bar chart

2. **Churn by Gender:** Bar chart

3. **Churn by Education Level:** Bar chart

4. **Churn by Employment Status:** Bar chart

5. **Churn by Number of Products:** Bar chart

6. **Churn by Digital Banking Usage:** Bar chart (0, 1, or 2 channels)

7. **Churn by Loan Status:** Bar chart

### Page 3: Churn Drivers
**Purpose:** Identify behavioral factors associated with churn

**Visuals:**
1. **Churn by Days Since Last Transaction:** Bar chart with recency buckets

2. **Churn by Number of Complaints:** Bar chart

3. **Churn by Digital Engagement Score:** Bar chart with score bands

4. **Churn by Average Monthly Balance:** Bar chart with balance quartiles

5. **Churn by Account Tenure:** Bar chart with tenure buckets

6. **Feature Importance Table:** Table showing top predictive features (from permutation importance):
   - Days_Since_Last_Transaction: 7.4%
   - Account_Tenure_Months: 5.3%
   - Number_of_Complaints: 4.1%
   - Number_of_Products: 2.9%
   - Average_Monthly_Deposit: 1.0%

### Page 4: High-Risk Customers
**Purpose:** Actionable list of customers at highest churn risk

**Visuals:**
1. **Risk Band Breakdown:** Donut chart

2. **High-Risk Customer Table:** Table with columns:
   - Customer_ID
   - Churn_Probability
   - Risk_Band
   - Days_Since_Last_Transaction
   - Number_of_Complaints
   - Number_of_Products
   - Digital_Engagement_Score
   - Customer_Segment
   - Branch_Location
   - Average_Monthly_Balance

3. **High-Risk by Segment:** Bar chart

4. **High-Risk by Branch:** Bar chart

5. **Slicer:** Risk Band selector

### Page 5: Capacity Analysis
**Purpose:** Understand trade-offs between outreach capacity and churn detection

**Visuals:**
1. **Capacity vs Precision:** Line chart showing precision at different capacity levels (100, 500, 1000, 1500, 2000 customers)
2. **Capacity vs Recall:** Line chart showing recall at different capacity levels
3. **Capacity vs F1:** Line chart showing F1 score at different capacity levels
4. **Capacity Table:** Table with capacity scenarios:
   - Capacity 100: 67% precision, 10.7% recall
   - Capacity 500: 39% precision, 31% recall
   - Capacity 1000: 29% precision, 47% recall
   - Capacity 2000: 22% precision, 69% recall

### Page 6: Threshold Analysis
**Purpose:** Understand impact of different decision thresholds

**Visuals:**
1. **Threshold vs Metrics:** Line chart showing precision, recall, and F1 across thresholds (0.30 to 0.70)
2. **Threshold Table:** Table with threshold analysis:
   - Threshold 0.30: 15.8% precision, 92.7% recall, 73.5% customers flagged
   - Threshold 0.50: 22.7% precision, 65.3% recall, 36.1% customers flagged
   - Threshold 0.70: 36.9% precision, 32.8% recall, 11.2% customers flagged
3. **Confusion Matrix at 0.50:** Matrix showing TP, FP, TN, FN

## Key Findings to Highlight

### Finding 1: Customer Value vs Churn Risk
**Finding:** Customer value (balance) and churn risk are largely independent. High-balance customers are not significantly less likely to churn than average-balance customers. This means retention efforts should not be limited to low-value customers.

**Visualization:** Scatter plot showing Average Monthly Balance vs Churn Probability, with risk band color coding.

### Finding 2: Recency is Strongest Predictor
**Finding:** Days since last transaction is the strongest predictor of churn (7.4% permutation importance). Customers inactive for 90+ days have significantly higher churn rates.

**Visualization:** Bar chart showing churn rate by recency buckets.

### Finding 3: Model vs Baseline
**Finding:** The model achieves 65.3% recall on churned customers compared to 0% for the baseline (which always predicts retention). This means the model can identify 65% of customers who will churn, enabling proactive retention.

**Visualization:** Comparison table of model vs baseline metrics.

### Finding 4: Risk Bands are Well-Calibrated
**Finding:** Risk bands are well-calibrated on held-out test data:
- Low Risk: 3.9% observed churn rate
- Medium Risk: 12.1% observed churn rate
- High Risk: 33.1% observed churn rate

**Visualization:** Bar chart showing observed churn rate by risk band.

## Implementation Notes

1. **Data Refresh:** The dashboard should be set to refresh when the underlying CSV files are updated (e.g., after retraining the model with `python scripts/train_pipeline.py`).

2. **Row-Level Security:** Not applicable for this synthetic dataset, but in a real banking environment, implement RLS to restrict access by branch or region.

3. **Performance:** With 20,000 rows, performance should be acceptable. For larger datasets, consider using Power BI Premium or incremental refresh.

4. **Drill-Through:** Enable drill-through from the High-Risk Customer table to the Customer Profile page for detailed analysis.

5. **Model Metadata:** The model performance metrics (ROC-AUC: 0.74, calibration ECE: 0.31) should be displayed as static values or loaded from the model metadata.

## Data Export for Power BI

Run the following command to generate the Power BI data exports:
```bash
python scripts/train_pipeline.py
```

This will create:
- `data/processed/churn_predictions.csv` - Customer predictions with risk bands
- `data/processed/churn_customers_clean.csv` - Cleaned feature data
- `models/logistic_regression_pipeline.joblib` - Trained model with metadata

## Limitations

- This is a specification only; the actual Power BI .pbix file is not included in this repository.
- The dashboard uses synthetic data for educational purposes.
- Model metrics and risk bands are based on a specific model version; retraining may change these values.
- Geographic coordinates for Malawian branches are not provided; a map visualization would require manual geocoding.
- The model has an Expected Calibration Error (ECE) of 0.31, indicating probabilities are not perfectly calibrated. Consider this when setting operational thresholds.
- Permutation importance shows association, not causation. High importance does not mean changing the feature will reduce churn. 
8. Screenshot into `visuals/dashboard/` and link from the main README
