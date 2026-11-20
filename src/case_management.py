"""Portable retention case-register utilities. CSV is the persistence hand-off."""
from datetime import date
import pandas as pd

CASE_STATUSES = ["New", "Assigned", "Contact attempted", "Follow-up required", "Retained", "Closed - churned", "Closed - not proceeding"]
CASE_COLUMNS = ["Case_ID", "Customer_ID", "Case_Status", "Assigned_Manager", "Next_Follow_Up", "Last_Contact_Date", "Contact_Channel", "Latest_Outcome", "Resolution_Reason", "Case_Notes"]

def build_starter_case_register(customers: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    required = {"Customer_ID", "Churn_Probability"}
    missing = required - set(customers.columns)
    if missing:
        raise ValueError("Customer data is missing: " + ", ".join(sorted(missing)))
    eligible = customers.copy()
    if "Risk_Band" in eligible:
        eligible = eligible[eligible["Risk_Band"].isin(["High Risk", "Medium Risk"])]
    eligible = eligible.sort_values("Churn_Probability", ascending=False).drop_duplicates("Customer_ID").head(limit)
    rows = pd.DataFrame({
        "Case_ID": [f"RET-{i:05d}" for i in range(1, len(eligible) + 1)],
        "Customer_ID": eligible["Customer_ID"].astype(str).values,
        "Case_Status": "New", "Assigned_Manager": "", "Next_Follow_Up": "",
        "Last_Contact_Date": "", "Contact_Channel": "", "Latest_Outcome": "",
        "Resolution_Reason": "", "Case_Notes": "",
    })
    return rows[CASE_COLUMNS]

def validate_case_register(register: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(register, pd.DataFrame):
        raise TypeError("Case register must be a pandas DataFrame.")
    missing = set(CASE_COLUMNS) - set(register.columns)
    if missing:
        raise ValueError("Case register is missing required columns: " + ", ".join(sorted(missing)))
    result = register[CASE_COLUMNS].copy()
    for col in ["Case_ID", "Customer_ID"]:
        result[col] = result[col].fillna("").astype(str).str.strip()
        if result[col].eq("").any():
            raise ValueError(f"{col} cannot be blank.")
        if result[col].duplicated().any():
            raise ValueError(f"{col} values must be unique.")
    result["Case_Status"] = result["Case_Status"].fillna("New").astype(str).str.strip()
    invalid = sorted(set(result["Case_Status"]) - set(CASE_STATUSES))
    if invalid:
        raise ValueError("Invalid case status: " + ", ".join(invalid))
    for col in ["Assigned_Manager", "Contact_Channel", "Latest_Outcome", "Resolution_Reason", "Case_Notes"]:
        result[col] = result[col].fillna("").astype(str).str.strip()
    for col in ["Next_Follow_Up", "Last_Contact_Date"]:
        raw = result[col].fillna("").astype(str).str.strip()
        parsed = pd.to_datetime(raw.replace("", pd.NA), errors="coerce")
        if (raw.ne("") & parsed.isna()).any():
            raise ValueError(f"{col} contains an invalid date; use YYYY-MM-DD.")
        result[col] = parsed.dt.strftime("%Y-%m-%d").fillna("")
    return result

def add_case_monitoring_fields(register: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    today = today or date.today()
    result = register.copy()
    follow_up = pd.to_datetime(result["Next_Follow_Up"], errors="coerce")
    result["Case_Open"] = result["Case_Status"].isin(CASE_STATUSES[:4])
    result["Follow_Up_Overdue"] = result["Case_Open"] & follow_up.notna() & (follow_up.dt.date < today)
    return result
