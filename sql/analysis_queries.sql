-- ============================================================================
-- analysis_queries.sql
-- Analytical queries for the Banking Customer Churn dataset (customers table
-- in sql/churn_analysis.db, built from data/processed/churn_customers_clean.csv).
-- Every query answers a specific business question -- none included purely
-- for demonstration.
-- ============================================================================

-- Q1: Overall customer churn rate.
-- Business purpose: the single headline number for any churn report.
SELECT
    COUNT(*) AS total_customers,
    SUM(Churn) AS churned_customers,
    ROUND(AVG(Churn), 4) AS churn_rate
FROM customers;


-- Q2: Churn rate by branch.
-- Business purpose: identifies which branches need retention attention first.
SELECT
    Branch_Location,
    COUNT(*) AS customers,
    ROUND(AVG(Churn), 4) AS churn_rate
FROM customers
GROUP BY Branch_Location
ORDER BY churn_rate DESC;


-- Q3: Churn rate by customer segment.
-- Business purpose: segments are how a bank organizes products and relationship
-- management, so this is the most actionable "who" cut of the churn problem.
SELECT
    Customer_Segment,
    COUNT(*) AS customers,
    ROUND(AVG(Churn), 4) AS churn_rate
FROM customers
GROUP BY Customer_Segment
ORDER BY churn_rate DESC;


-- Q4: Average balance by churn status.
-- Business purpose: checks whether churners are disproportionately low-value
-- or high-value customers -- both are actionable, but for different reasons.
SELECT
    CASE WHEN Churn = 1 THEN 'Churned' ELSE 'Retained' END AS status,
    ROUND(AVG(Average_Monthly_Balance), 2) AS avg_balance,
    COUNT(*) AS customers
FROM customers
GROUP BY status;


-- Q5: Average transaction activity by churn status.
-- Business purpose: confirms the EDA finding (lower activity, higher churn)
-- directly in SQL, as a cross-check independent of the Python analysis.
SELECT
    CASE WHEN Churn = 1 THEN 'Churned' ELSE 'Retained' END AS status,
    ROUND(AVG(Average_Monthly_Transactions), 2) AS avg_monthly_transactions,
    ROUND(AVG(Days_Since_Last_Transaction), 2) AS avg_days_since_last_txn,
    COUNT(*) AS customers
FROM customers
GROUP BY status;


-- Q6: Customers with high complaint counts (2 or more), and their churn rate.
-- Business purpose: a direct worklist for customer service follow-up --
-- not just a statistic, but the actual set of customers to act on.
SELECT
    Customer_ID, Branch_Location, Customer_Segment, Number_of_Complaints,
    Number_of_Products, Days_Since_Last_Transaction, Churn
FROM customers
WHERE Number_of_Complaints >= 2
ORDER BY Number_of_Complaints DESC
LIMIT 20;


-- Q6b: Churn rate specifically among high-complaint customers, for context
-- against the overall churn rate from Q1.
SELECT
    CASE WHEN Number_of_Complaints >= 2 THEN '2+ complaints' ELSE '0-1 complaints' END AS complaint_group,
    COUNT(*) AS customers,
    ROUND(AVG(Churn), 4) AS churn_rate
FROM customers
GROUP BY complaint_group;


-- Q7: Customers inactive for more than 90 days.
-- Business purpose: a concrete, actionable inactivity worklist -- the
-- threshold (90 days) is a stated, adjustable business parameter, not a
-- hard-coded assumption.
SELECT
    Customer_ID, Branch_Location, Customer_Segment, Days_Since_Last_Transaction,
    Average_Monthly_Balance, Number_of_Products, Churn
FROM customers
WHERE Days_Since_Last_Transaction > 90
ORDER BY Days_Since_Last_Transaction DESC
LIMIT 20;


-- Q7b: Churn rate for customers inactive >90 days vs. everyone else.
SELECT
    CASE WHEN Days_Since_Last_Transaction > 90 THEN 'Inactive 90+ days' ELSE 'Active within 90 days' END AS activity_group,
    COUNT(*) AS customers,
    ROUND(AVG(Churn), 4) AS churn_rate
FROM customers
GROUP BY activity_group;


-- Q8: High-value customers (top balance quartile) with elevated churn risk.
-- Business purpose: "higher-value customers should not automatically have
-- zero churn risk" -- this query finds exactly the customers who prove that,
-- combining a CTE (for the quartile cutoff) with a WHERE filter.
WITH balance_quartile AS (
    SELECT
        *,
        NTILE(4) OVER (ORDER BY Average_Monthly_Balance) AS balance_qtile
    FROM customers
)
SELECT
    Customer_ID, Branch_Location, Customer_Segment, Average_Monthly_Balance,
    Number_of_Complaints, Days_Since_Last_Transaction, Churn
FROM balance_quartile
WHERE balance_qtile = 4 AND Churn = 1
ORDER BY Average_Monthly_Balance DESC
LIMIT 20;


-- Q8b: Churn rate within the high-value (top balance quartile) segment specifically.
-- Business purpose: quantifies exactly how "not zero" the high-value churn
-- risk actually is, rather than leaving it as an anecdote from Q8's list.
WITH balance_quartile AS (
    SELECT *, NTILE(4) OVER (ORDER BY Average_Monthly_Balance) AS balance_qtile
    FROM customers
)
SELECT
    CASE WHEN balance_qtile = 4 THEN 'Top balance quartile' ELSE 'Other 75%' END AS value_group,
    COUNT(*) AS customers,
    ROUND(AVG(Churn), 4) AS churn_rate
FROM balance_quartile
GROUP BY value_group;


-- Q9: Digital banking adoption vs. churn, using a CASE-based engagement label.
-- Business purpose: translates the raw 0/1/2 channel count into a readable
-- label a relationship manager report can use directly.
SELECT
    CASE
        WHEN Has_Mobile_Banking = 1 AND Has_Internet_Banking = 1 THEN 'Fully digital'
        WHEN Has_Mobile_Banking = 1 OR Has_Internet_Banking = 1 THEN 'Partially digital'
        ELSE 'Not digital'
    END AS digital_engagement_group,
    COUNT(*) AS customers,
    ROUND(AVG(Churn), 4) AS churn_rate
FROM customers
GROUP BY digital_engagement_group
ORDER BY churn_rate DESC;


-- Q10: Product count vs. churn, with a window-function rank of segments by churn rate.
-- Business purpose: ranks segments so management can prioritize attention
-- without eyeballing a table of five numbers.
WITH segment_churn AS (
    SELECT
        Customer_Segment,
        COUNT(*) AS customers,
        AVG(Churn) AS churn_rate
    FROM customers
    GROUP BY Customer_Segment
)
SELECT
    Customer_Segment, customers, ROUND(churn_rate, 4) AS churn_rate,
    RANK() OVER (ORDER BY churn_rate DESC) AS churn_risk_rank
FROM segment_churn
ORDER BY churn_risk_rank;
