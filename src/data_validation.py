"""
Data validation and quality checks for the banking churn dataset.

This module provides functions to validate schema, data types, ranges,
and detect data quality issues in the synthetic customer churn dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Container for validation results."""
    passed: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, int]


class DataValidator:
    """Validates banking customer churn dataset against expected schema and constraints."""
    
    # Expected schema
    REQUIRED_COLUMNS = [
        "Customer_ID", "Age", "Gender", "Marital_Status", "Education_Level",
        "Occupation", "Employment_Status", "Monthly_Income", "Account_Tenure_Months",
        "Customer_Segment", "Branch_Location", "Number_of_Products", "Has_Credit_Card",
        "Has_Mobile_Banking", "Has_Internet_Banking", "Average_Monthly_Balance",
        "Average_Monthly_Transactions", "Total_Transaction_Value_Last_6_Months",
        "Number_of_Complaints", "Days_Since_Last_Transaction", "Average_Monthly_Deposit",
        "Average_Monthly_Withdrawal", "Loan_Status", "Has_Active_Loan",
        "Digital_Engagement_Score", "Customer_Service_Interactions", "Churn"
    ]
    
    # Valid categorical values
    VALID_GENDERS = ["Male", "Female"]
    VALID_MARITAL_STATUS = ["Single", "Married", "Divorced", "Widowed"]
    VALID_EDUCATION = ["Primary", "Secondary", "Tertiary/College", "University"]
    VALID_SEGMENTS = ["Retail", "SME", "Premium", "Salary Account", "Student/Youth"]
    VALID_BRANCHES = [
        "Blantyre", "Lilongwe", "Mzuzu", "Zomba", "Mangochi",
        "Kasungu", "Salima", "Liwonde", "Rumphi", "Karonga"
    ]
    VALID_LOAN_STATUS = ["No Active Loan", "Active", "Active - Good Standing", "Active - Past Due"]
    
    # Numeric ranges
    AGE_RANGE = (18, 90)
    TENURE_RANGE = (1, 240)
    PRODUCTS_RANGE = (1, 6)
    TRANSACTIONS_RANGE = (0, 60)
    COMPLAINTS_RANGE = (0, 10)
    DAYS_SINCE_RANGE = (0, 400)
    DIGITAL_SCORE_RANGE = (0, 100)
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize validator with dataframe.
        
        Args:
            df: DataFrame to validate
        """
        self.df = df.copy()
        self.errors = []
        self.warnings = []
        self.summary = {}
    
    def validate_schema(self) -> bool:
        """Check that all required columns are present."""
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        if missing_cols:
            self.errors.append(f"Missing required columns: {missing_cols}")
            return False
        
        extra_cols = set(self.df.columns) - set(self.REQUIRED_COLUMNS)
        if extra_cols:
            self.warnings.append(f"Extra columns present: {extra_cols}")
        
        return True
    
    def validate_data_types(self) -> bool:
        """Check that columns have appropriate data types."""
        passed = True
        
        # Numeric columns
        numeric_cols = [
            "Age", "Monthly_Income", "Account_Tenure_Months", "Number_of_Products",
            "Has_Credit_Card", "Has_Mobile_Banking", "Has_Internet_Banking",
            "Average_Monthly_Balance", "Average_Monthly_Transactions",
            "Total_Transaction_Value_Last_6_Months", "Number_of_Complaints",
            "Days_Since_Last_Transaction", "Average_Monthly_Deposit",
            "Average_Monthly_Withdrawal", "Has_Active_Loan",
            "Digital_Engagement_Score", "Customer_Service_Interactions", "Churn"
        ]
        
        for col in numeric_cols:
            if col in self.df.columns:
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    self.errors.append(f"Column {col} should be numeric but is {self.df[col].dtype}")
                    passed = False
        
        # Categorical columns
        categorical_cols = [
            "Customer_ID", "Gender", "Marital_Status", "Education_Level",
            "Occupation", "Employment_Status", "Customer_Segment",
            "Branch_Location", "Loan_Status"
        ]
        
        for col in categorical_cols:
            if col in self.df.columns:
                if col != "Customer_ID" and not pd.api.types.is_object_dtype(self.df[col]):
                    self.warnings.append(f"Column {col} expected to be object/string type")
        
        return passed
    
    def validate_ranges(self) -> bool:
        """Check that numeric values are within expected ranges."""
        passed = True
        
        # Age
        if "Age" in self.df.columns:
            invalid_age = ((self.df["Age"] < self.AGE_RANGE[0]) | 
                          (self.df["Age"] > self.AGE_RANGE[1])).sum()
            if invalid_age > 0:
                self.errors.append(f"{invalid_age} records with Age outside valid range {self.AGE_RANGE}")
                passed = False
        
        # Account Tenure
        if "Account_Tenure_Months" in self.df.columns:
            invalid_tenure = ((self.df["Account_Tenure_Months"] < self.TENURE_RANGE[0]) | 
                             (self.df["Account_Tenure_Months"] > self.TENURE_RANGE[1])).sum()
            if invalid_tenure > 0:
                self.errors.append(f"{invalid_tenure} records with Account_Tenure_Months outside valid range {self.TENURE_RANGE}")
                passed = False
        
        # Number of Products
        if "Number_of_Products" in self.df.columns:
            invalid_products = ((self.df["Number_of_Products"] < self.PRODUCTS_RANGE[0]) | 
                               (self.df["Number_of_Products"] > self.PRODUCTS_RANGE[1])).sum()
            if invalid_products > 0:
                self.errors.append(f"{invalid_products} records with Number_of_Products outside valid range {self.PRODUCTS_RANGE}")
                passed = False
        
        # Transactions (should not be negative)
        if "Average_Monthly_Transactions" in self.df.columns:
            negative_txn = (self.df["Average_Monthly_Transactions"] < 0).sum()
            if negative_txn > 0:
                self.errors.append(f"{negative_txn} records with negative Average_Monthly_Transactions")
                passed = False
        
        # Digital Engagement Score
        if "Digital_Engagement_Score" in self.df.columns:
            invalid_score = ((self.df["Digital_Engagement_Score"] < self.DIGITAL_SCORE_RANGE[0]) | 
                           (self.df["Digital_Engagement_Score"] > self.DIGITAL_SCORE_RANGE[1])).sum()
            if invalid_score > 0:
                self.errors.append(f"{invalid_score} records with Digital_Engagement_Score outside valid range {self.DIGITAL_SCORE_RANGE}")
                passed = False
        
        # Churn (should be 0 or 1)
        if "Churn" in self.df.columns:
            invalid_churn = (~self.df["Churn"].isin([0, 1])).sum()
            if invalid_churn > 0:
                self.errors.append(f"{invalid_churn} records with Churn values other than 0 or 1")
                passed = False
        
        return passed
    
    def validate_categorical_values(self) -> bool:
        """Check that categorical columns contain only valid values."""
        passed = True
        
        if "Gender" in self.df.columns:
            invalid_gender = (~self.df["Gender"].isin(self.VALID_GENDERS)).sum()
            if invalid_gender > 0:
                self.errors.append(f"{invalid_gender} records with invalid Gender values")
                passed = False
        
        if "Customer_Segment" in self.df.columns:
            invalid_segment = (~self.df["Customer_Segment"].isin(self.VALID_SEGMENTS)).sum()
            if invalid_segment > 0:
                self.errors.append(f"{invalid_segment} records with invalid Customer_Segment values")
                passed = False
        
        if "Branch_Location" in self.df.columns:
            invalid_branch = (~self.df["Branch_Location"].isin(self.VALID_BRANCHES)).sum()
            if invalid_branch > 0:
                self.errors.append(f"{invalid_branch} records with invalid Branch_Location values")
                passed = False
        
        if "Loan_Status" in self.df.columns:
            invalid_loan = (~self.df["Loan_Status"].isin(self.VALID_LOAN_STATUS)).sum()
            if invalid_loan > 0:
                self.errors.append(f"{invalid_loan} records with invalid Loan_Status values")
                passed = False
        
        return passed
    
    def check_duplicates(self) -> bool:
        """Check for duplicate records."""
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.errors.append(f"Found {duplicates} duplicate records")
            return False
        return True
    
    def check_missing_values(self) -> bool:
        """Check for missing values and report them."""
        missing = self.df.isnull().sum()
        missing_cols = missing[missing > 0]
        
        if len(missing_cols) > 0:
            self.warnings.append(f"Missing values found: {missing_cols.to_dict()}")
            self.summary["missing_values"] = missing_cols.to_dict()
            return False  # Warning, not error
        
        return True
    
    def check_target_balance(self) -> bool:
        """Check target class distribution."""
        if "Churn" not in self.df.columns:
            return False
        
        churn_rate = self.df["Churn"].mean()
        self.summary["churn_rate"] = churn_rate
        self.summary["n_churned"] = int(self.df["Churn"].sum())
        self.summary["n_retained"] = int((self.df["Churn"] == 0).sum())
        
        if churn_rate < 0.01 or churn_rate > 0.50:
            self.warnings.append(f"Unusual churn rate: {churn_rate:.2%}")
        
        return True
    
    def validate_all(self) -> ValidationResult:
        """
        Run all validation checks.
        
        Returns:
            ValidationResult with all errors, warnings, and summary
        """
        self.validate_schema()
        self.validate_data_types()
        self.validate_ranges()
        self.validate_categorical_values()
        self.check_duplicates()
        self.check_missing_values()
        self.check_target_balance()
        
        passed = len(self.errors) == 0
        
        return ValidationResult(
            passed=passed,
            errors=self.errors,
            warnings=self.warnings,
            summary=self.summary
        )
    
    def generate_report(self) -> str:
        """Generate a human-readable validation report."""
        result = self.validate_all()
        
        report = ["=" * 60]
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Status: {'PASSED' if result.passed else 'FAILED'}")
        report.append(f"Total records: {len(self.df)}")
        report.append(f"Total columns: {len(self.df.columns)}")
        report.append("")
        
        if result.summary:
            report.append("SUMMARY:")
            for key, value in result.summary.items():
                report.append(f"  {key}: {value}")
            report.append("")
        
        if result.errors:
            report.append(f"ERRORS ({len(result.errors)}):")
            for error in result.errors:
                report.append(f"  - {error}")
            report.append("")
        
        if result.warnings:
            report.append(f"WARNINGS ({len(result.warnings)}):")
            for warning in result.warnings:
                report.append(f"  - {warning}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def validate_dataset(df: pd.DataFrame) -> ValidationResult:
    """
    Convenience function to validate a dataset.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        ValidationResult object
    """
    validator = DataValidator(df)
    return validator.validate_all()
