# SQL Analysis

A SQLite database built from the cleaned dataset, plus 13 analytical queries answering the business questions specified for this project.

## Files

- **`churn_analysis.db`** — generated from `data/processed/churn_customers_clean.csv` (single `customers` table; see note below on why this stayed single-table)
- **`analysis_queries.sql`** — 13 queries covering churn rate, branch/segment breakdowns, balance/activity comparisons, complaint and inactivity worklists, high-value-customer risk, digital engagement, and a window-function segment ranking

## Why a single table here, unlike the other portfolio projects

Earlier SQL layers in this portfolio (health facility data quality, medicine stock-outs, hospital patient flow) split a flat file into multiple related tables to demonstrate JOINs, because those datasets had genuine natural sub-entities (products, departments, doctors). This customer dataset is inherently one-row-per-customer with no repeating sub-entity to normalize out — forcing an artificial JOIN here would demonstrate the syntax without a real reason for it. Instead, this file leans on **CTEs and window functions** (`NTILE()`, `RANK()`) for its relational-thinking demonstration, which fit the data's actual shape.

## How to reproduce

```bash
cd sql
python3 -c "
import pandas as pd, sqlite3
df = pd.read_csv('../data/processed/churn_customers_clean.csv')
conn = sqlite3.connect('churn_analysis.db')
df.to_sql('customers', conn, if_exists='replace', index=False)
"
sqlite3 churn_analysis.db < analysis_queries.sql
```

## Query techniques demonstrated

- `GROUP BY`, `CASE`, `WHERE`, `ORDER BY` — all queries
- CTEs (`WITH`) — Q8, Q8b (balance quartile), Q10 (segment ranking)
- Window functions — `NTILE(4)` (Q8/Q8b, balance quartiles), `RANK()` (Q10, segment churn ranking)

## What each query answers

1. Overall churn rate — the headline number
2–3. Churn by branch / segment — "which groups need attention"
4–5. Balance / activity by churn status — cross-checks the Python EDA independently in SQL
6–7. High-complaint / long-inactive customer worklists — actual customer lists, not just statistics
8. High-value customers who *did* churn, quantified against the brief's point that high value doesn't mean zero risk — **the data shows high-value churn is not zero, but it's also not meaningfully elevated (12.53% vs. 12.55% overall)**, a nuance worth stating precisely rather than assuming the more dramatic version of the finding
9. Digital engagement banding vs. churn
10. Segment ranking by churn rate, via window function
