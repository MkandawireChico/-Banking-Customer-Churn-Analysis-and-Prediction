"""
Tests for data validation module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_validation import DataValidator, validate_dataset


@pytest.fixture
def sample_valid_data():
    """Create a sample valid dataset."""
    return pd.DataFrame({
        "Customer_ID": ["CUST-0000001", "CUST-0000002", "CUST-0000003"],
        "Age": [30, 45, 60],
        "Gender": ["Male", "Female", "Male"],
        "Marital_Status": ["Single", "Married", "Married"],
        "Education_Level": ["Secondary", "University", "Primary"],
        "Occupation": ["Employed - Private", "Business Owner", "Employed - Government"],
        "Employment_Status": ["Employed - Private", "Business Owner", "Employed - Government"],
        "Monthly_Income": [300000, 500000, 400000],
        "Account_Tenure_Months": [12, 48, 120],
        "Customer_Segment": ["Retail", "SME", "Premium"],
        "Branch_Location": ["Blantyre", "Lilongwe", "Mzuzu"],
        "Number_of_Products": [2, 3, 4],
        "Has_Credit_Card": [0, 1, 1],
        "Has_Mobile_Banking": [1, 1, 0],
        "Has_Internet_Banking": [1, 0, 1],
        "Average_Monthly_Balance": [200000, 500000, 1000000],
        "Average_Monthly_Transactions": [5, 10, 15],
        "Total_Transaction_Value_Last_6_Months": [1000000, 3000000, 5000000],
        "Number_of_Complaints": [0, 1, 0],
        "Days_Since_Last_Transaction": [10, 30, 60],
        "Average_Monthly_Deposit": [150000, 300000, 500000],
        "Average_Monthly_Withdrawal": [100000, 200000, 400000],
        "Loan_Status": ["No Active Loan", "Active", "Active - Good Standing"],
        "Has_Active_Loan": [0, 1, 1],
        "Digital_Engagement_Score": [50, 70, 80],
        "Customer_Service_Interactions": [0, 2, 1],
        "Churn": [0, 1, 0],
    })


def test_validate_schema_valid(sample_valid_data):
    """Test schema validation with valid data."""
    validator = DataValidator(sample_valid_data)
    assert validator.validate_schema() == True


def test_validate_schema_missing_column(sample_valid_data):
    """Test schema validation with missing column."""
    df = sample_valid_data.drop(columns=["Age"])
    validator = DataValidator(df)
    result = validator.validate_schema()
    assert result == False
    assert len(validator.errors) > 0


def test_validate_ranges_valid(sample_valid_data):
    """Test range validation with valid data."""
    validator = DataValidator(sample_valid_data)
    assert validator.validate_ranges() == True


def test_validate_ranges_invalid_age(sample_valid_data):
    """Test range validation with invalid age."""
    sample_valid_data.loc[0, "Age"] = 150
    validator = DataValidator(sample_valid_data)
    result = validator.validate_ranges()
    assert result == False
    assert any("Age" in error for error in validator.errors)


def test_validate_ranges_negative_transactions(sample_valid_data):
    """Test range validation with negative transactions."""
    sample_valid_data.loc[0, "Average_Monthly_Transactions"] = -5
    validator = DataValidator(sample_valid_data)
    result = validator.validate_ranges()
    assert result == False
    assert any("negative" in error.lower() for error in validator.errors)


def test_validate_categorical_valid(sample_valid_data):
    """Test categorical validation with valid data."""
    validator = DataValidator(sample_valid_data)
    assert validator.validate_categorical_values() == True


def test_validate_categorical_invalid_segment(sample_valid_data):
    """Test categorical validation with invalid segment."""
    sample_valid_data.loc[0, "Customer_Segment"] = "Invalid Segment"
    validator = DataValidator(sample_valid_data)
    result = validator.validate_categorical_values()
    assert result == False
    assert any("Customer_Segment" in error for error in validator.errors)


def test_check_duplicates_no_duplicates(sample_valid_data):
    """Test duplicate check with no duplicates."""
    validator = DataValidator(sample_valid_data)
    assert validator.check_duplicates() == True


def test_check_duplicates_with_duplicates(sample_valid_data):
    """Test duplicate check with duplicates."""
    df = pd.concat([sample_valid_data, sample_valid_data.head(1)], ignore_index=True)
    validator = DataValidator(df)
    result = validator.check_duplicates()
    assert result == False
    assert "duplicate" in validator.errors[0].lower()


def test_validate_all_passes(sample_valid_data):
    """Test full validation with valid data."""
    validator = DataValidator(sample_valid_data)
    result = validator.validate_all()
    assert result.passed == True
    assert len(result.errors) == 0


def test_validate_all_fails(sample_valid_data):
    """Test full validation with invalid data."""
    sample_valid_data.loc[0, "Age"] = 150
    sample_valid_data.loc[1, "Average_Monthly_Transactions"] = -5
    validator = DataValidator(sample_valid_data)
    result = validator.validate_all()
    assert result.passed == False
    assert len(result.errors) > 0


def test_validate_dataset_convenience_function(sample_valid_data):
    """Test convenience function validate_dataset."""
    result = validate_dataset(sample_valid_data)
    assert result.passed == True


def test_generate_report(sample_valid_data):
    """Test report generation."""
    validator = DataValidator(sample_valid_data)
    report = validator.generate_report()
    assert "DATA VALIDATION REPORT" in report
    assert "Status: PASSED" in report
