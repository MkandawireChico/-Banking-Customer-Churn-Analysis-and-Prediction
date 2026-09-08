# Power BI Dashboard Guide

**Status: not yet built.** Specification written from the actual fields and results in `data/processed/churn_customers_clean.csv` and `data/processed/churn_predictions.csv`. All data is synthetic — see the project README's disclaimer.

---

## 1. Data sources

- `data/processed/churn_customers_clean.csv` — full customer base
- `data/processed/churn_predictions.csv` — test-set predictions (`Customer_ID`, `Actual_Churn`, `Churn_Probability`, `Predicted_Churn`, `Risk_Band`) for Page 4

## 2. Data model

Single flat customer table plus the predictions table, joined on `Customer_ID`. No date dimension is needed — this is a cross-sectional snapshot, not a time series (see Limitations in the README).

## 3. Measures (DAX)

```dax
Total Customers = COUNTROWS(customers)
Churned Customers = SUM(customers[Churn])
Overall Churn Rate = DIVIDE([Churned Customers], [Total Customers])
Average Churn Probability = AVERAGE(predictions[Churn_Probability])
High Risk Customers = CALCULATE(COUNTROWS(predictions), predictions[Risk_Band] = "High Risk")
Average Tenure = AVERAGE(customers[Account_Tenure_Months])
Average Products per Customer = AVERAGE(customers[Number_of_Products])
```

**Why `DIVIDE()`:** returns `BLANK()` rather than erroring under a zero-row filter context — relevant once Branch/Segment/Risk-Band slicers are applied.

## 4. Pages

### Page 1 — Executive Overview
KPIs: Total Customers, Churned Customers, Overall Churn Rate, Average Churn Probability, High-Risk Customers, Average Tenure, Average Products per Customer.
Visuals: churn by customer segment (bar), churn by branch (bar), churned-vs-retained (donut). *(No genuine time trend is available — see Limitations; a "trend" visual would need a real or simulated date dimension added in a future version.)*

### Page 2 — Customer Profile
Churn by age group, gender, income band, customer segment, tenure band, product count, loan status, digital banking adoption — all bar charts, matching the notebook's Section 7 EDA exactly.

### Page 3 — Churn Drivers
Top churn predictors (table, from `coef_df`/`importance_df` exported to CSV), complaints vs. churn, transaction activity vs. churn, digital engagement vs. churn, products vs. churn, tenure vs. churn, days since last transaction vs. churn.

### Page 4 — High-Risk Customers
Table: `Customer_ID`, `Branch`, `Customer_Segment`, `Tenure`, `Number_of_Products`, `Transaction_Activity`, `Complaints`, `Days_Since_Last_Transaction`, `Churn_Probability`, `Risk_Band` — filterable by Branch, Customer Segment, Risk Band, Age Group, Digital Banking, Product Count.

This page should let a Relationship Manager answer, directly and without extra clicks: *"Who should I contact first?"* (sort by `Churn_Probability` descending within High Risk), *"Why are they high risk?"* (the complaint/activity/tenure columns sit right next to the probability), *"Which groups have the highest churn?"* (Page 2/3), *"Which branches need attention?"* (Page 1).

## 5. A finding that belongs on the dashboard, not just the README

**High-value customers are not meaningfully more or less likely to churn than anyone else** (12.53% in the top balance quartile vs. 12.55% overall — see `sql/analysis_queries.sql` Q8b). This is worth a specific callout on Page 1 or Page 2, because the instinctive assumption ("our best customers are safe" or the opposite, "our best customers are most at risk") is *wrong either way* in this data — value and churn risk are close to independent here, which changes how a retention budget should be allocated (spread by *risk band*, not by *account value*).

## 6. Build order

1. Import both data sources, relate on `Customer_ID`
2. Write core measures (Section 3)
3. Build Page 1 — sense-check every KPI against the notebook's printed Section 14 summary
4. Build Pages 2–3
5. Build Page 4, sorted by probability within risk band by default
6. Add the Section 5 callout explicitly
7. Add slicers last
8. Screenshot into `visuals/dashboard/` and link from the main README
