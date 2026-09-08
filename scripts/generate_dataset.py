"""
Generates the synthetic banking customer churn dataset for the portfolio
project. ALL DATA IS ENTIRELY SYNTHETIC — see the project README's
disclaimer. "Nyanja Commercial Bank" is fictional.

Run:
    python generate_dataset.py
Outputs:
    ../data/raw/churn_customers_synthetic.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 20000

BRANCHES = ["Blantyre", "Lilongwe", "Mzuzu", "Zomba", "Mangochi",
            "Kasungu", "Salima", "Liwonde", "Rumphi", "Karonga"]
BRANCH_WEIGHTS = [0.22, 0.22, 0.12, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05]
SEGMENTS = ["Retail", "SME", "Premium", "Salary Account", "Student/Youth"]
SEGMENT_WEIGHTS = [0.35, 0.15, 0.10, 0.30, 0.10]
EDUCATION = ["Primary", "Secondary", "Tertiary/College", "University"]


def employment_for_segment(seg):
    if seg == "Salary Account":
        return np.random.choice(["Employed - Government", "Employed - Private"], p=[0.45, 0.55])
    if seg == "SME":
        return "Business Owner"
    if seg == "Premium":
        return np.random.choice(["Employed - Private", "Business Owner"], p=[0.7, 0.3])
    if seg == "Student/Youth":
        return np.random.choice(["Student", "Employed - Private", "Unemployed"], p=[0.6, 0.25, 0.15])
    return np.random.choice(["Employed - Private", "Self-Employed", "Unemployed"], p=[0.45, 0.35, 0.20])


def generate_clean_dataset():
    age = np.clip(np.random.normal(37, 12, N), 18, 75).round().astype(int)
    gender = np.random.choice(["Male", "Female"], N, p=[0.53, 0.47])
    marital_status = np.random.choice(["Single", "Married", "Divorced", "Widowed"], N, p=[0.32, 0.53, 0.10, 0.05])
    education = np.random.choice(EDUCATION, N, p=[0.10, 0.35, 0.30, 0.25])
    segment = np.random.choice(SEGMENTS, N, p=SEGMENT_WEIGHTS)
    employment_status = np.array([employment_for_segment(s) for s in segment])
    branch = np.random.choice(BRANCHES, N, p=BRANCH_WEIGHTS)

    income_base = {"Retail": 280000, "SME": 550000, "Premium": 1100000,
                    "Salary Account": 320000, "Student/Youth": 120000}
    monthly_income = np.array([
        max(50000, np.random.lognormal(np.log(income_base[s]), 0.45)) for s in segment
    ]).round(-2)

    account_tenure_months = np.clip(np.random.exponential(40, N), 1, 240).round().astype(int)
    number_of_products = np.clip(np.random.poisson(1 + account_tenure_months / 60, N), 1, 6)

    has_credit_card = np.random.binomial(1, np.where(segment == "Premium", 0.65, 0.20))
    has_mobile_banking = np.random.binomial(1, np.clip(0.9 - age / 200, 0.3, 0.9))
    has_internet_banking = np.random.binomial(1, np.clip(0.7 - age / 250, 0.15, 0.75))

    digital_engagement_score = np.clip(
        30 + has_mobile_banking * 30 + has_internet_banking * 20 + np.random.normal(0, 12, N), 0, 100
    ).round(1)

    avg_monthly_transactions = np.clip(np.random.poisson(4 + digital_engagement_score / 10, N), 0, 60)
    avg_monthly_deposit = (monthly_income * np.random.uniform(0.3, 0.9, N)).round(-2)
    avg_monthly_withdrawal = (avg_monthly_deposit * np.random.uniform(0.5, 1.05, N)).round(-2)
    avg_monthly_balance = np.clip(
        (avg_monthly_deposit - avg_monthly_withdrawal) * np.random.uniform(2, 6, N) +
        monthly_income * np.random.uniform(0.2, 1.5, N), 0, None
    ).round(-2)

    total_transaction_value_6mo = (avg_monthly_transactions * (avg_monthly_deposit + avg_monthly_withdrawal) / 2 * 6 *
                                     np.random.uniform(0.8, 1.2, N)).round(-2)
    days_since_last_transaction = np.clip(
        np.random.exponential(30 - avg_monthly_transactions * 0.8 + 5, N), 0, 400
    ).round().astype(int)

    number_of_complaints = np.random.poisson(0.3 + (100 - digital_engagement_score) / 200, N)
    customer_service_interactions = number_of_complaints + np.random.poisson(0.5, N)

    has_active_loan = np.random.binomial(1, np.where(segment == "SME", 0.5, 0.25))
    loan_status = np.where(
        has_active_loan == 1,
        np.random.choice(["Active", "Active - Good Standing", "Active - Past Due"], N, p=[0.5, 0.35, 0.15]),
        "No Active Loan"
    )

    balance_to_income_ratio = avg_monthly_balance / monthly_income
    z = (
        -0.85
        + 0.020 * days_since_last_transaction
        - 0.35 * np.log1p(avg_monthly_transactions)
        + 0.55 * number_of_complaints
        - 0.35 * number_of_products
        - 0.012 * account_tenure_months
        - 0.018 * digital_engagement_score / 10
        - 0.15 * has_mobile_banking
        - 0.10 * has_internet_banking
        - 0.20 * np.log1p(balance_to_income_ratio)
        + 0.10 * (loan_status == "Active - Past Due").astype(int)
        - 0.05 * (loan_status == "Active - Good Standing").astype(int)
        + 0.10 * (segment == "Student/Youth").astype(int)
        - 0.08 * (segment == "Premium").astype(int)
        + np.random.normal(0, 0.6, N)
    )
    prob_churn = 1 / (1 + np.exp(-z))
    churn = np.random.binomial(1, prob_churn)

    return pd.DataFrame({
        "Customer_ID": [f"CUST-{i:07d}" for i in range(1, N + 1)],
        "Age": age, "Gender": gender, "Marital_Status": marital_status,
        "Education_Level": education, "Occupation": employment_status, "Employment_Status": employment_status,
        "Monthly_Income": monthly_income, "Account_Tenure_Months": account_tenure_months,
        "Customer_Segment": segment, "Branch_Location": branch,
        "Number_of_Products": number_of_products, "Has_Credit_Card": has_credit_card,
        "Has_Mobile_Banking": has_mobile_banking, "Has_Internet_Banking": has_internet_banking,
        "Average_Monthly_Balance": avg_monthly_balance, "Average_Monthly_Transactions": avg_monthly_transactions,
        "Total_Transaction_Value_Last_6_Months": total_transaction_value_6mo,
        "Number_of_Complaints": number_of_complaints, "Days_Since_Last_Transaction": days_since_last_transaction,
        "Average_Monthly_Deposit": avg_monthly_deposit, "Average_Monthly_Withdrawal": avg_monthly_withdrawal,
        "Loan_Status": loan_status, "Has_Active_Loan": has_active_loan,
        "Digital_Engagement_Score": digital_engagement_score,
        "Customer_Service_Interactions": customer_service_interactions,
        "Churn": churn,
    })


def inject_data_quality_issues(df):
    df = df.copy()
    idx = df.sample(frac=0.02, random_state=10).index
    df.loc[idx, "Monthly_Income"] = np.nan

    idx = df.sample(frac=0.015, random_state=11).index
    df.loc[idx, "Marital_Status"] = np.nan

    dup_idx = df.sample(frac=0.01, random_state=20).index
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    idx = df.sample(frac=0.015, random_state=30).index
    segment_variant_map = {
        "Retail": "retail", "SME": "S.M.E", "Premium": "PREMIUM",
        "Salary Account": "Salary account", "Student/Youth": "Student / Youth",
    }
    for i in idx:
        original = df.loc[i, "Customer_Segment"]
        if original in segment_variant_map:
            df.loc[i, "Customer_Segment"] = segment_variant_map[original]

    idx = df.sample(frac=0.002, random_state=40).index
    df.loc[idx, "Age"] = np.random.choice([5, 8, 130, 150, -3], len(idx))

    idx = df.sample(frac=0.003, random_state=50).index
    df.loc[idx, "Monthly_Income"] = df.loc[idx, "Monthly_Income"] * np.random.uniform(10, 20, len(idx))

    idx = df.sample(frac=0.003, random_state=60).index
    df.loc[idx, "Average_Monthly_Transactions"] = -np.abs(df.loc[idx, "Average_Monthly_Transactions"]) - 1

    idx = df.sample(frac=0.015, random_state=70).index
    df.loc[idx, "Has_Internet_Banking"] = np.nan

    return df


def main():
    df_clean = generate_clean_dataset()
    print(f"Clean dataset generated: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")
    print(f"Overall churn rate (before quality-issue injection): {df_clean['Churn'].mean():.2%}")

    df_raw = inject_data_quality_issues(df_clean)
    print(f"Raw dataset after injecting quality issues: {df_raw.shape[0]:,} rows")

    out_dir = Path("../data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "churn_customers_synthetic.csv"
    df_raw.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
