"""Reproducible banking churn dataset generator. Run from any working directory."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N = 20000
BRANCHES = ["Blantyre", "Lilongwe", "Mzuzu", "Zomba", "Mangochi", "Kasungu", "Salima", "Liwonde", "Rumphi", "Karonga"]
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


def generate_clean_dataset(random_state=42, n=N):
    """Generate a reproducible underlying customer population."""
    np.random.seed(random_state)
    age = np.clip(np.random.normal(37, 12, n), 18, 75).round().astype(int)
    gender = np.random.choice(["Male", "Female"], n, p=[0.53, 0.47])
    marital_status = np.random.choice(["Single", "Married", "Divorced", "Widowed"], n, p=[0.32, 0.53, 0.10, 0.05])
    education = np.random.choice(EDUCATION, n, p=[0.10, 0.35, 0.30, 0.25])
    segment = np.random.choice(SEGMENTS, n, p=SEGMENT_WEIGHTS)
    employment_status = np.array([employment_for_segment(s) for s in segment])
    branch = np.random.choice(BRANCHES, n, p=BRANCH_WEIGHTS)
    income_base = {"Retail": 280000, "SME": 550000, "Premium": 1100000, "Salary Account": 320000, "Student/Youth": 120000}
    monthly_income = np.array([max(50000, np.random.lognormal(np.log(income_base[s]), 0.45)) for s in segment]).round(-2)
    tenure = np.clip(np.random.exponential(40, n), 1, 240).round().astype(int)
    products = np.clip(np.random.poisson(1 + tenure / 60, n), 1, 6)
    credit = np.random.binomial(1, np.where(segment == "Premium", 0.65, 0.20))
    mobile = np.random.binomial(1, np.clip(0.9 - age / 200, 0.3, 0.9))
    internet = np.random.binomial(1, np.clip(0.7 - age / 250, 0.15, 0.75))
    digital = np.clip(30 + mobile * 30 + internet * 20 + np.random.normal(0, 12, n), 0, 100).round(1)
    transactions = np.clip(np.random.poisson(4 + digital / 10, n), 0, 60)
    deposits = (monthly_income * np.random.uniform(0.3, 0.9, n)).round(-2)
    withdrawals = (deposits * np.random.uniform(0.5, 1.05, n)).round(-2)
    balance = np.clip((deposits - withdrawals) * np.random.uniform(2, 6, n) + monthly_income * np.random.uniform(0.2, 1.5, n), 0, None).round(-2)
    txn_value = (transactions * (deposits + withdrawals) / 2 * 6 * np.random.uniform(0.8, 1.2, n)).round(-2)
    recency = np.clip(np.random.exponential(30 - transactions * 0.8 + 5, n), 0, 400).round().astype(int)
    complaints = np.random.poisson(0.3 + (100 - digital) / 200, n)
    interactions = complaints + np.random.poisson(0.5, n)
    active_loan = np.random.binomial(1, np.where(segment == "SME", 0.5, 0.25))
    loan_status = np.where(active_loan == 1, np.random.choice(["Active", "Active - Good Standing", "Active - Past Due"], n, p=[0.5, 0.35, 0.15]), "No Active Loan")
    balance_income = balance / monthly_income
    z = (-0.85 + 0.020 * recency - 0.35 * np.log1p(transactions) + 0.55 * complaints
         - 0.35 * products - 0.012 * tenure - 0.018 * digital / 10 - 0.15 * mobile - 0.10 * internet
         - 0.20 * np.log1p(balance_income) + 0.10 * (loan_status == "Active - Past Due").astype(int)
         - 0.05 * (loan_status == "Active - Good Standing").astype(int)
         + 0.10 * (segment == "Student/Youth").astype(int) - 0.08 * (segment == "Premium").astype(int)
         + np.random.normal(0, 0.6, n))
    churn = np.random.binomial(1, 1 / (1 + np.exp(-z)))
    return pd.DataFrame({
        "Customer_ID": [f"CUST-{i:07d}" for i in range(1, n + 1)], "Age": age, "Gender": gender,
        "Marital_Status": marital_status, "Education_Level": education, "Occupation": employment_status,
        "Employment_Status": employment_status, "Monthly_Income": monthly_income,
        "Account_Tenure_Months": tenure, "Customer_Segment": segment, "Branch_Location": branch,
        "Number_of_Products": products, "Has_Credit_Card": credit, "Has_Mobile_Banking": mobile,
        "Has_Internet_Banking": internet, "Average_Monthly_Balance": balance,
        "Average_Monthly_Transactions": transactions, "Total_Transaction_Value_Last_6_Months": txn_value,
        "Number_of_Complaints": complaints, "Days_Since_Last_Transaction": recency,
        "Average_Monthly_Deposit": deposits, "Average_Monthly_Withdrawal": withdrawals,
        "Loan_Status": loan_status, "Has_Active_Loan": active_loan,
        "Digital_Engagement_Score": digital, "Customer_Service_Interactions": interactions, "Churn": churn,
    })


def inject_data_quality_issues(df, random_state=42):
    """Inject repeatable missing values, duplicates, inconsistent labels and outliers."""
    rng = np.random.RandomState(random_state)
    df = df.copy()
    idx = df.sample(frac=0.02, random_state=10).index
    df.loc[idx, "Monthly_Income"] = np.nan
    idx = df.sample(frac=0.015, random_state=11).index
    df.loc[idx, "Marital_Status"] = np.nan
    dup_idx = df.sample(frac=0.01, random_state=20).index
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)
    idx = df.sample(frac=0.015, random_state=30).index
    variants = {"Retail": "retail", "SME": "S.M.E", "Premium": "PREMIUM", "Salary Account": "Salary account", "Student/Youth": "Student / Youth"}
    for i in idx:
        value = df.loc[i, "Customer_Segment"]
        if value in variants:
            df.loc[i, "Customer_Segment"] = variants[value]
    idx = df.sample(frac=0.002, random_state=40).index
    df.loc[idx, "Age"] = rng.choice([5, 8, 130, 150, -3], len(idx))
    idx = df.sample(frac=0.003, random_state=50).index
    df.loc[idx, "Monthly_Income"] = df.loc[idx, "Monthly_Income"] * rng.uniform(10, 20, len(idx))
    idx = df.sample(frac=0.003, random_state=60).index
    df.loc[idx, "Average_Monthly_Transactions"] = -np.abs(df.loc[idx, "Average_Monthly_Transactions"]) - 1
    idx = df.sample(frac=0.015, random_state=70).index
    df.loc[idx, "Has_Internet_Banking"] = np.nan
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate the banking churn dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=N)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "raw" / "churn_customers_synthetic.csv")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be at least 1")
    data = inject_data_quality_issues(generate_clean_dataset(args.seed, args.rows), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Generated {len(data):,} records across {len(data.columns)} columns")
    print(f"Overall churn rate: {data['Churn'].mean():.2%}")
    print(f"Saved dataset: {args.output.resolve()}")


if __name__ == "__main__":
    main()
