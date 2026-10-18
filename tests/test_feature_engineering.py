"""
Tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer, get_feature_definitions, check_division_by_zero_risks


@pytest.fixture
def sample_data():
    """Create sample data for feature engineering."""
    return pd.DataFrame({
        "Age": [30, 45, 60],
        "Monthly_Income": [300000, 500000, 400000],
        "Account_Tenure_Months": [12, 48, 120],
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
        "Has_Active_Loan": [0, 1, 1],
        "Digital_Engagement_Score": [50, 70, 80],
        "Customer_Service_Interactions": [0, 2, 1],
    })


def test_feature_engineer_init():
    """Test FeatureEngineer initialization."""
    fe = FeatureEngineer()
    assert fe.is_fitted_ == False
    assert fe.balance_q75_ is None
    assert fe.txn_min_ is None
    assert fe.txn_max_ is None


def test_feature_engineer_fit(sample_data):
    """Test FeatureEngineer fit."""
    fe = FeatureEngineer()
    fe.fit(sample_data)
    assert fe.is_fitted_ == True
    assert fe.balance_q75_ is not None
    assert fe.txn_min_ is not None
    assert fe.txn_max_ is not None


def test_feature_engineer_transform_without_fit(sample_data):
    """Test that transform raises error if not fitted."""
    fe = FeatureEngineer()
    with pytest.raises(ValueError, match="must be fitted"):
        fe.transform(sample_data)


def test_feature_engineer_transform(sample_data):
    """Test FeatureEngineer transform."""
    fe = FeatureEngineer()
    fe.fit(sample_data)
    result = fe.transform(sample_data)
    
    # Check that engineered features are present
    assert "Average_Transaction_Value" in result.columns
    assert "Transaction_Activity_Index" in result.columns
    assert "Digital_Banking_Usage" in result.columns
    assert "Complaint_Rate" in result.columns
    assert "Engagement_Index" in result.columns
    assert "Balance_to_Income_Ratio" in result.columns
    assert "Products_per_Tenure" in result.columns
    assert "Recent_Activity_Flag" in result.columns
    assert "High_Value_Customer_Flag" in result.columns
    assert "Low_Engagement_Flag" in result.columns


def test_feature_engineer_fit_transform(sample_data):
    """Test fit_transform pattern."""
    fe = FeatureEngineer()
    result = fe.fit(sample_data).transform(sample_data)
    
    assert len(result) == len(sample_data)
    assert "Average_Transaction_Value" in result.columns


def test_average_transaction_value_calculation(sample_data):
    """Test Average_Transaction_Value calculation."""
    fe = FeatureEngineer()
    fe.fit(sample_data)
    result = fe.transform(sample_data)
    
    # Manual calculation for first row
    expected = 1000000 / (5 * 6)
    assert abs(result.iloc[0]["Average_Transaction_Value"] - expected) < 1.0


def test_transaction_activity_index_calculation(sample_data):
    """Test Transaction_Activity_Index calculation."""
    fe = FeatureEngineer()
    fe.fit(sample_data)
    result = fe.transform(sample_data)
    
    # Manual calculation for first row
    expected = 5 / (10 + 1)
    assert abs(result.iloc[0]["Transaction_Activity_Index"] - expected) < 0.01


def test_digital_banking_usage_calculation(sample_data):
    """Test Digital_Banking_Usage calculation."""
    fe = FeatureEngineer()
    fe.fit(sample_data)
    result = fe.transform(sample_data)
    
    # First row: mobile=1, internet=1 -> 2
    assert result.iloc[0]["Digital_Banking_Usage"] == 2
    # Second row: mobile=1, internet=0 -> 1
    assert result.iloc[1]["Digital_Banking_Usage"] == 1


def test_recent_activity_flag(sample_data):
    """Test Recent_Activity_Flag calculation."""
    fe = FeatureEngineer()
    fe.fit(sample_data)
    result = fe.transform(sample_data)
    
    # First row: 10 days -> should be 1
    assert result.iloc[0]["Recent_Activity_Flag"] == 1
    # Third row: 60 days -> should be 0
    assert result.iloc[2]["Recent_Activity_Flag"] == 0


def test_high_value_customer_flag(sample_data):
    """Test High_Value_Customer_Flag calculation."""
    fe = FeatureEngineer()
    fe.fit(sample_data)
    result = fe.transform(sample_data)
    
    # Third row has highest balance (1M) -> should be flagged as high value
    assert result.iloc[2]["High_Value_Customer_Flag"] == 1


def test_low_engagement_flag(sample_data):
    """Test Low_Engagement_Flag calculation."""
    fe = FeatureEngineer()
    fe.fit(sample_data)
    result = fe.transform(sample_data)
    
    # All scores are >= 40, so none should be low engagement
    assert (result["Low_Engagement_Flag"] == 0).all()


def test_get_feature_definitions():
    """Test get_feature_definitions function."""
    definitions = get_feature_definitions()
    
    assert isinstance(definitions, dict)
    assert len(definitions) == 10
    assert "Average_Transaction_Value" in definitions
    assert "Transaction_Activity_Index" in definitions
    assert "Digital_Banking_Usage" in definitions
    
    # Check that each definition has required keys
    for feature, info in definitions.items():
        assert "formula" in info
        assert "meaning" in info
        assert "availability" in info


def test_check_division_by_zero_risks(sample_data):
    """Test check_division_by_zero_risks function."""
    risks = check_division_by_zero_risks(sample_data)
    
    assert isinstance(risks, dict)
    # All transactions are positive, so no risk
    assert risks["Average_Transaction_Value"] == 0
    assert risks["Complaint_Rate"] == 0
    assert risks["Products_per_Tenure"] == 0


def test_check_division_by_zero_risks_with_zero_transactions(sample_data):
    """Test check_division_by_zero_risks with zero transactions."""
    sample_data.loc[0, "Average_Monthly_Transactions"] = 0
    risks = check_division_by_zero_risks(sample_data)
    
    assert risks["Average_Transaction_Value"] == 1


def test_check_division_by_zero_risks_with_zero_tenure(sample_data):
    """Test check_division_by_zero_risks with zero tenure."""
    sample_data.loc[0, "Account_Tenure_Months"] = 0
    risks = check_division_by_zero_risks(sample_data)
    
    assert risks["Complaint_Rate"] == 1
    assert risks["Products_per_Tenure"] == 1


def test_get_feature_names_out():
    """Test get_feature_names_out method."""
    fe = FeatureEngineer()
    feature_names = fe.get_feature_names_out()
    
    assert len(feature_names) == 10
    assert "Average_Transaction_Value" in feature_names
