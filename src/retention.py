"""
Retention recommendation framework for banking customer churn.

This module provides:
- Risk band validation
- Reason code generation
- Retention action recommendations
- Outreach queue prioritization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class RetentionRecommender:
    """
    Generates retention recommendations based on model predictions
    and customer profiles.
    """
    
    def __init__(self):
        """Initialize retention recommender."""
        self.risk_band_thresholds = {
            'Low Risk': (0.0, 0.33),
            'Medium Risk': (0.33, 0.66),
            'High Risk': (0.66, 1.0)
        }
    
    def assign_risk_band(self, probability: float) -> str:
        """
        Assign risk band based on churn probability.
        
        Args:
            probability: Churn probability
            
        Returns:
            Risk band label
        """
        for band, (low, high) in self.risk_band_thresholds.items():
            if low <= probability < high:
                return band
        return 'High Risk'  # Default for probability >= 0.66
    
    def validate_risk_bands(
        self,
        df: pd.DataFrame,
        probability_col: str = 'Churn_Probability',
        actual_col: str = 'Actual_Churn'
    ) -> pd.DataFrame:
        """
        Validate risk bands on held-out data.
        
        This checks whether risk bands actually separate customers
        by their observed churn rates.
        
        Args:
            df: DataFrame with predictions and actual outcomes
            probability_col: Column with churn probabilities
            actual_col: Column with actual churn labels
            
        Returns:
            DataFrame with validation metrics per band
        """
        df = df.copy()
        df['Risk_Band'] = df[probability_col].apply(self.assign_risk_band)
        
        validation_results = []
        
        for band in ['Low Risk', 'Medium Risk', 'High Risk']:
            band_data = df[df['Risk_Band'] == band]
            
            if len(band_data) > 0:
                results = {
                    'risk_band': band,
                    'n_customers': len(band_data),
                    'share_of_total': len(band_data) / len(df),
                    'avg_probability': band_data[probability_col].mean(),
                    'observed_churn_rate': band_data[actual_col].mean(),
                    'n_churned': int(band_data[actual_col].sum()),
                }
                
                # Precision and recall if we treat this band as positive
                if band == 'High Risk':
                    tp = ((band_data[actual_col] == 1) & (band_data['Risk_Band'] == band)).sum()
                    fp = ((band_data[actual_col] == 0) & (band_data['Risk_Band'] == band)).sum()
                    fn = ((df[actual_col] == 1) & (df['Risk_Band'] != band)).sum()
                    
                    results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
                    results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                    results['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                
                validation_results.append(results)
        
        return pd.DataFrame(validation_results)
    
    def generate_reason_codes(
        self,
        customer_data: pd.Series,
        probability: float
    ) -> List[str]:
        """
        Generate reason codes for a customer's churn risk.
        
        These are rule-based explanations based on the customer's
        profile, not model-specific feature importance.
        
        Args:
            customer_data: Series with customer features
            probability: Churn probability
            
        Returns:
            List of reason codes
        """
        reasons = []
        
        # Ensure data is a Series
        data = customer_data if isinstance(customer_data, pd.Series) else pd.Series(customer_data)
        
        # Recency-based reasons
        if 'Days_Since_Last_Transaction' in data.index:
            days = data['Days_Since_Last_Transaction']
            if days > 90:
                reasons.append("High inactivity - no transactions in 90+ days")
            elif days > 30:
                reasons.append("Moderate inactivity - no transactions in 30+ days")
        
        # Complaint-based reasons
        if 'Number_of_Complaints' in data.index:
            complaints = data['Number_of_Complaints']
            if complaints >= 2:
                reasons.append(f"Multiple complaints ({complaints}) - unresolved service issues")
            elif complaints == 1:
                reasons.append("Recent complaint - service dissatisfaction")
        
        # Product adoption reasons
        if 'Number_of_Products' in data.index:
            products = data['Number_of_Products']
            if products == 1:
                reasons.append("Single product - low relationship depth")
            elif products >= 4:
                reasons.append(f"Multiple products ({products}) - strong relationship")
        
        # Digital engagement reasons
        if 'Digital_Engagement_Score' in data.index:
            score = data['Digital_Engagement_Score']
            if score < 40:
                reasons.append(f"Low digital engagement (score: {score:.1f})")
            elif score > 70:
                reasons.append(f"High digital engagement (score: {score:.1f})")
        
        # Tenure reasons
        if 'Account_Tenure_Months' in data.index:
            tenure = data['Account_Tenure_Months']
            if tenure < 12:
                reasons.append(f"New customer ({tenure} months) - early relationship stage")
            elif tenure > 60:
                reasons.append(f"Long-term customer ({tenure} months) - established relationship")
        
        # Transaction activity reasons
        if 'Average_Monthly_Transactions' in data.index:
            txn = data['Average_Monthly_Transactions']
            if txn < 2:
                reasons.append("Very low transaction activity")
            elif txn > 15:
                reasons.append("High transaction activity")
        
        # Balance reasons
        if 'Average_Monthly_Balance' in data.index:
            balance = data['Average_Monthly_Balance']
            if balance > 1000000:
                reasons.append("High balance customer - retention priority")
        
        # Loan status reasons
        if 'Loan_Status' in data.index:
            loan = data['Loan_Status']
            if loan == 'Active - Past Due':
                reasons.append("Loan past due - financial stress indicator")
        
        return reasons
    
    def recommend_action(
        self,
        customer_data: pd.Series,
        probability: float,
        risk_band: str
    ) -> Dict[str, str]:
        """
        Recommend retention action based on risk profile.
        
        Args:
            customer_data: Customer feature data (Series)
            probability: Churn probability
            risk_band: Assigned risk band
            
        Returns:
            Dictionary with recommended action and priority
        """
        reasons = self.generate_reason_codes(customer_data, probability)
        
        # Ensure data is a Series
        data = customer_data if isinstance(customer_data, pd.Series) else pd.Series(customer_data)
        
        high_inactivity = 'Days_Since_Last_Transaction' in data.index and data['Days_Since_Last_Transaction'] > 90
        multiple_complaints = 'Number_of_Complaints' in data.index and data['Number_of_Complaints'] >= 2
        low_products = 'Number_of_Products' in data.index and data['Number_of_Products'] == 1
        low_digital = 'Digital_Engagement_Score' in data.index and data['Digital_Engagement_Score'] < 40
        high_balance = 'Average_Monthly_Balance' in data.index and data['Average_Monthly_Balance'] > 1000000
        
        # Determine action based on profile
        if risk_band == 'High Risk':
            if high_balance:
                action = "Priority relationship manager check-in - high-value customer at risk"
                priority = "Critical"
            elif multiple_complaints:
                action = "Customer service intervention - resolve outstanding complaints before retention offer"
                priority = "High"
            elif high_inactivity:
                action = "Relationship manager outreach - re-engagement campaign"
                priority = "High"
            elif low_products:
                action = "Product education and cross-sell opportunity"
                priority = "Medium"
            elif low_digital:
                action = "Digital banking support session - improve channel adoption"
                priority = "Medium"
            else:
                action = "Standard retention outreach - personalized offer"
                priority = "Medium"
        
        elif risk_band == 'Medium Risk':
            if high_inactivity:
                action = "Lighter-touch digital re-engagement campaign"
                priority = "Low"
            elif multiple_complaints:
                action = "Service quality check - follow up on recent complaints"
                priority = "Medium"
            else:
                action = "Monitor - no immediate action needed"
                priority = "Low"
        
        else:  # Low Risk
            action = "No action needed - maintain relationship"
            priority = "None"
        
        return {
            'recommended_action': action,
            'priority': priority,
            'reason_codes': reasons
        }
    
    def create_outreach_queue(
        self,
        df: pd.DataFrame,
        capacity: Optional[int] = None,
        probability_col: str = 'Churn_Probability',
        customer_id_col: str = 'Customer_ID'
    ) -> pd.DataFrame:
        """
        Create a ranked outreach queue with capacity constraints.
        
        Args:
            df: DataFrame with predictions and customer data
            capacity: Maximum number of customers to include (None = all high risk)
            probability_col: Column with churn probabilities
            customer_id_col: Column with customer IDs
            
        Returns:
            DataFrame with ranked outreach queue
        """
        df = df.copy()
        df['Risk_Band'] = df[probability_col].apply(self.assign_risk_band)
        
        # Filter to high and medium risk if capacity is specified
        if capacity is not None:
            df = df[df['Risk_Band'].isin(['High Risk', 'Medium Risk'])]
        
        # Sort by probability descending
        df = df.sort_values(probability_col, ascending=False)
        
        # Apply capacity constraint
        if capacity is not None and len(df) > capacity:
            df = df.head(capacity)
        
        # Add rank
        df['Outreach_Rank'] = range(1, len(df) + 1)
        
        # Generate recommendations for each customer
        recommendations = []
        for idx, row in df.iterrows():
            rec = self.recommend_action(
                row.drop([probability_col, 'Risk_Band', customer_id_col, 'Outreach_Rank']),
                row[probability_col],
                row['Risk_Band']
            )
            recommendations.append(rec)
        
        rec_df = pd.DataFrame(recommendations)
        df = pd.concat([df.reset_index(drop=True), rec_df], axis=1)
        
        # Reorder columns for readability
        priority_order = ['Outreach_Rank', customer_id_col, 'Risk_Band', probability_col,
                        'priority', 'recommended_action', 'reason_codes']
        other_cols = [col for col in df.columns if col not in priority_order]
        df = df[priority_order + other_cols]
        
        return df


def generate_retention_report(
    df: pd.DataFrame,
    probability_col: str = 'Churn_Probability',
    actual_col: str = 'Actual_Churn',
    customer_id_col: str = 'Customer_ID'
) -> str:
    """
    Generate a comprehensive retention analysis report.
    
    Args:
        df: DataFrame with predictions and actual outcomes
        probability_col: Column with churn probabilities
        actual_col: Column with actual churn labels
        customer_id_col: Column with customer IDs
        
    Returns:
        String report
    """
    recommender = RetentionRecommender()
    
    report = ["=" * 60]
    report.append("RETENTION ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Risk band validation
    report.append("RISK BAND VALIDATION")
    report.append("-" * 60)
    validation_df = recommender.validate_risk_bands(df, probability_col, actual_col)
    report.append(validation_df.to_string(index=False))
    report.append("")
    
    # Check if risk increases across bands
    if len(validation_df) >= 3:
        low_rate = validation_df[validation_df['risk_band'] == 'Low Risk']['observed_churn_rate'].values[0]
        med_rate = validation_df[validation_df['risk_band'] == 'Medium Risk']['observed_churn_rate'].values[0]
        high_rate = validation_df[validation_df['risk_band'] == 'High Risk']['observed_churn_rate'].values[0]
        
        if low_rate < med_rate < high_rate:
            report.append("✓ Risk bands are well-calibrated - churn rate increases across bands")
        else:
            report.append("⚠ Risk bands may need adjustment - churn rate does not increase monotonically")
    report.append("")
    
    # Outreach queue statistics
    report.append("OUTREACH QUEUE STATISTICS")
    report.append("-" * 60)
    high_risk = (df[probability_col] >= 0.66).sum()
    medium_risk = ((df[probability_col] >= 0.33) & (df[probability_col] < 0.66)).sum()
    low_risk = (df[probability_col] < 0.33).sum()
    
    report.append(f"High Risk customers: {high_risk} ({high_risk/len(df):.1%})")
    report.append(f"Medium Risk customers: {medium_risk} ({medium_risk/len(df):.1%})")
    report.append(f"Low Risk customers: {low_risk} ({low_risk/len(df):.1%})")
    report.append("")
    
    # Capacity scenarios
    report.append("CAPACITY SCENARIOS")
    report.append("-" * 60)
    for capacity in [100, 500, 1000]:
        queue = recommender.create_outreach_queue(df, capacity=capacity)
        if len(queue) > 0:
            avg_prob = queue[probability_col].mean()
            expected_churners = (queue[probability_col] >= 0.66).sum()
            report.append(f"Capacity {capacity}: {len(queue)} customers, avg prob {avg_prob:.3f}, ~{expected_churners} high-risk")
    report.append("")
    
    report.append("=" * 60)
    report.append("IMPORTANT NOTES")
    report.append("=" * 60)
    report.append("- Risk bands summarize model scores, they do not guarantee future outcomes.")
    report.append("- Recommendations are hypotheses for retention teams to evaluate.")
    report.append("- This analysis uses synthetic data for educational purposes.")
    report.append("- Real retention interventions should be tested with controlled experiments.")
    report.append("- Do not claim financial savings or ROI without actual intervention data.")
    report.append("=" * 60)
    
    return "\n".join(report)
