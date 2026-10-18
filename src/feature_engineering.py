"""
Feature engineering for banking customer churn prediction.

This module provides functions to create engineered features from the raw dataset.
All transformations are designed to be fit on training data only to prevent leakage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer that fits on training data and transforms
    both training and test data using only training statistics.
    
    This prevents data leakage by ensuring that normalization thresholds
    (like min/max for scaling, quartiles for flags) are computed only from
    the training set.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.balance_q75_ = None
        self.txn_min_ = None
        self.txn_max_ = None
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureEngineer':
        """
        Fit feature engineering parameters on training data.
        
        Args:
            X: Training DataFrame
            y: Target (ignored, present for scikit-learn API compatibility)
            
        Returns:
            self
        """
        # Store quartile threshold for High_Value_Customer_Flag
        if "Average_Monthly_Balance" in X.columns:
            self.balance_q75_ = X["Average_Monthly_Balance"].quantile(0.75)
        
        # Store min/max for transaction normalization (Engagement_Index)
        # This fixes the leakage issue in the original notebook
        if "Average_Monthly_Transactions" in X.columns:
            self.txn_min_ = X["Average_Monthly_Transactions"].min()
            self.txn_max_ = X["Average_Monthly_Transactions"].max()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted parameters.
        
        Args:
            X: DataFrame to transform
            
        Returns:
            Transformed DataFrame with engineered features
        """
        if not self.is_fitted_:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        X = X.copy()
        
        # Feature 1: Average_Transaction_Value
        # Formula: Total_Transaction_Value_Last_6_Months / (Average_Monthly_Transactions * 6)
        # Banking meaning: typical transaction size
        if "Total_Transaction_Value_Last_6_Months" in X.columns and "Average_Monthly_Transactions" in X.columns:
            X["Average_Transaction_Value"] = X["Total_Transaction_Value_Last_6_Months"] / (
                X["Average_Monthly_Transactions"].clip(lower=0.5) * 6
            )
        
        # Feature 2: Transaction_Activity_Index
        # Formula: Average_Monthly_Transactions / (Days_Since_Last_Transaction + 1)
        # Banking meaning: combined activity and recency signal
        if "Average_Monthly_Transactions" in X.columns and "Days_Since_Last_Transaction" in X.columns:
            X["Transaction_Activity_Index"] = X["Average_Monthly_Transactions"] / (
                X["Days_Since_Last_Transaction"] + 1
            )
        
        # Feature 3: Digital_Banking_Usage
        # Formula: Has_Mobile_Banking + Has_Internet_Banking (0, 1, or 2)
        # Banking meaning: breadth of digital channel adoption
        if "Has_Mobile_Banking" in X.columns and "Has_Internet_Banking" in X.columns:
            X["Digital_Banking_Usage"] = X["Has_Mobile_Banking"].fillna(0) + X["Has_Internet_Banking"].fillna(0)
        
        # Feature 4: Complaint_Rate
        # Formula: Number_of_Complaints / (Account_Tenure_Months / 12)
        # Banking meaning: complaints per year of relationship
        if "Number_of_Complaints" in X.columns and "Account_Tenure_Months" in X.columns:
            X["Complaint_Rate"] = X["Number_of_Complaints"] / (
                X["Account_Tenure_Months"] / 12
            ).clip(lower=0.1)
        
        # Feature 5: Engagement_Index
        # Formula: average of scaled Digital_Engagement_Score and normalized transaction activity
        # Banking meaning: composite engagement measure
        # CRITICAL FIX: Use fitted min/max from training data, not data being transformed
        if "Digital_Engagement_Score" in X.columns and "Average_Monthly_Transactions" in X.columns:
            if self.txn_min_ is not None and self.txn_max_ is not None and self.txn_max_ > self.txn_min_:
                norm_txn = (X["Average_Monthly_Transactions"] - self.txn_min_) / (
                    self.txn_max_ - self.txn_min_
                )
            else:
                norm_txn = pd.Series(0.5, index=X.index)  # Default if range is invalid
            
            X["Engagement_Index"] = ((X["Digital_Engagement_Score"] / 100) + norm_txn.fillna(0)) / 2
        
        # Feature 6: Balance_to_Income_Ratio
        # Formula: Average_Monthly_Balance / Monthly_Income
        # Banking meaning: how much of customer's financial life is with this bank
        if "Average_Monthly_Balance" in X.columns and "Monthly_Income" in X.columns:
            X["Balance_to_Income_Ratio"] = X["Average_Monthly_Balance"] / X["Monthly_Income"]
        
        # Feature 7: Products_per_Tenure
        # Formula: Number_of_Products / (Account_Tenure_Months / 12)
        # Banking meaning: rate of product adoption per year
        if "Number_of_Products" in X.columns and "Account_Tenure_Months" in X.columns:
            X["Products_per_Tenure"] = X["Number_of_Products"] / (
                X["Account_Tenure_Months"] / 12
            ).clip(lower=0.1)
        
        # Feature 8: Recent_Activity_Flag
        # Formula: 1 if Days_Since_Last_Transaction <= 30
        # Banking meaning: simple actionable recency cutoff
        if "Days_Since_Last_Transaction" in X.columns:
            X["Recent_Activity_Flag"] = (X["Days_Since_Last_Transaction"] <= 30).astype(int)
        
        # Feature 9: High_Value_Customer_Flag
        # Formula: 1 if Average_Monthly_Balance >= 75th percentile (from training data)
        # Banking meaning: flags customers whose departure matters most financially
        if "Average_Monthly_Balance" in X.columns and self.balance_q75_ is not None:
            X["High_Value_Customer_Flag"] = (X["Average_Monthly_Balance"] >= self.balance_q75_).astype(int)
        
        # Feature 10: Low_Engagement_Flag
        # Formula: 1 if Digital_Engagement_Score < 40
        # Banking meaning: simple actionable digital-engagement risk flag
        if "Digital_Engagement_Score" in X.columns:
            X["Low_Engagement_Flag"] = (X["Digital_Engagement_Score"] < 40).astype(int)
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for scikit-learn compatibility."""
        return [
            "Average_Transaction_Value",
            "Transaction_Activity_Index",
            "Digital_Banking_Usage",
            "Complaint_Rate",
            "Engagement_Index",
            "Balance_to_Income_Ratio",
            "Products_per_Tenure",
            "Recent_Activity_Flag",
            "High_Value_Customer_Flag",
            "Low_Engagement_Flag"
        ]


def get_feature_definitions() -> Dict[str, Dict[str, str]]:
    """
    Return documentation for all engineered features.
    
    Returns:
        Dictionary mapping feature names to their definitions and business meanings
    """
    return {
        "Average_Transaction_Value": {
            "formula": "Total_Transaction_Value_Last_6_Months / (Average_Monthly_Transactions * 6)",
            "meaning": "Typical size of each transaction - distinguishes frequent-small-value customers from infrequent-large-value ones",
            "edge_cases": "Division by zero prevented by clipping denominator to 0.5",
            "availability": "Available at prediction time (uses historical 6-month transaction data)"
        },
        "Transaction_Activity_Index": {
            "formula": "Average_Monthly_Transactions / (Days_Since_Last_Transaction + 1)",
            "meaning": "Combined signal of how active and how recent a customer's usage is",
            "edge_cases": "Days_Since_Last_Transaction = 0 handled by +1 in denominator",
            "availability": "Available at prediction time"
        },
        "Digital_Banking_Usage": {
            "formula": "Has_Mobile_Banking + Has_Internet_Banking (0, 1, or 2)",
            "meaning": "Breadth of digital channel adoption",
            "edge_cases": "Missing values treated as 0 (no channel)",
            "availability": "Available at prediction time"
        },
        "Complaint_Rate": {
            "formula": "Number_of_Complaints / (Account_Tenure_Months / 12)",
            "meaning": "Complaints per year of relationship - normalizes for relationship length",
            "edge_cases": "Tenure clipped at最低 0.1 years to prevent division by zero",
            "availability": "Available at prediction time"
        },
        "Engagement_Index": {
            "formula": "Average of (Digital_Engagement_Score / 100) and normalized transaction activity",
            "meaning": "Single composite engagement measure combining digital score and transaction frequency",
            "edge_cases": "Normalization uses training-set min/max to prevent leakage",
            "availability": "Available at prediction time"
        },
        "Balance_to_Income_Ratio": {
            "formula": "Average_Monthly_Balance / Monthly_Income",
            "meaning": "How much of a customer's financial life sits with this bank, relative to their means",
            "edge_cases": "Can be infinite if income is 0 (income has minimum floor in data generation)",
            "availability": "Available at prediction time"
        },
        "Products_per_Tenure": {
            "formula": "Number_of_Products / (Account_Tenure_Months / 12)",
            "meaning": "Rate of product adoption per year - distinguishes slow vs fast adopters",
            "edge_cases": "Tenure clipped at minimum 0.1 years",
            "availability": "Available at prediction time"
        },
        "Recent_Activity_Flag": {
            "formula": "1 if Days_Since_Last_Transaction <= 30, else 0",
            "meaning": "Simple, interpretable recency cutoff a relationship manager can act on directly",
            "edge_cases": "None - binary flag",
            "availability": "Available at prediction time"
        },
        "High_Value_Customer_Flag": {
            "formula": "1 if Average_Monthly_Balance >= 75th percentile (from training data)",
            "meaning": "Flags customers whose departure would matter most financially",
            "edge_cases": "Threshold computed from training data only to prevent leakage",
            "availability": "Available at prediction time (uses training-derived threshold)"
        },
        "Low_Engagement_Flag": {
            "formula": "1 if Digital_Engagement_Score < 40, else 0",
            "meaning": "Simple, actionable digital-engagement risk flag",
            "edge_cases": "None - binary flag with fixed threshold",
            "availability": "Available at prediction time"
        }
    }


def check_division_by_zero_risks(df: pd.DataFrame) -> Dict[str, int]:
    """
    Check for potential division-by-zero issues in feature calculations.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary of feature names to count of records at risk
    """
    risks = {}
    
    if "Average_Monthly_Transactions" in df.columns:
        risks["Average_Transaction_Value"] = (df["Average_Monthly_Transactions"] == 0).sum()
    
    if "Account_Tenure_Months" in df.columns:
        risks["Complaint_Rate"] = (df["Account_Tenure_Months"] == 0).sum()
        risks["Products_per_Tenure"] = (df["Account_Tenure_Months"] == 0).sum()
    
    if "Monthly_Income" in df.columns:
        risks["Balance_to_Income_Ratio"] = (df["Monthly_Income"] == 0).sum()
    
    return risks
