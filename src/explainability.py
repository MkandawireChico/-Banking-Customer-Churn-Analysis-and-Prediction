"""
Model explainability for banking customer churn prediction.

This module provides functions for:
- Permutation feature importance
- SHAP value explanations (optional dependency)
- Local explanations for individual predictions
- Coefficient interpretation for logistic regression
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator


class ModelExplainer:
    """Provides model explainability methods."""
    
    def __init__(self, model: BaseEstimator, feature_names: List[str]):
        """
        Initialize explainer.
        
        Args:
            model: Trained model pipeline
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
    
    def get_permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10,
        random_state: int = 42,
        scoring: str = 'roc_auc'
    ) -> pd.DataFrame:
        """
        Calculate permutation feature importance.
        
        Permutation importance measures how much the model's performance
        decreases when a feature's values are randomly shuffled. This is
        more reliable than model-specific importance metrics and works
        across different model types.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_repeats: Number of permutation repeats
            random_state: Random seed
            scoring: Scoring metric
            
        Returns:
            DataFrame with importance scores and std
        """
        result = permutation_importance(
            self.model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring
        )
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    def get_logistic_coefficients(self) -> Optional[pd.DataFrame]:
        """
        Extract logistic regression coefficients with interpretation.
        
        Returns:
            DataFrame with coefficients, odds ratios, and interpretation
        """
        # Try to access the logistic regression model from the pipeline
        if hasattr(self.model, 'named_steps'):
            if 'model' in self.model.named_steps:
                lr_model = self.model.named_steps['model']
                
                if hasattr(lr_model, 'coef_'):
                    coef = lr_model.coef_[0]
                    
                    # Get feature names after preprocessing
                    if hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
                        preprocessor = self.model.named_steps['preprocessor']
                        
                        # Get one-hot encoded feature names
                        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
                        cat_features = ohe.get_feature_names_out()
                        
                        # Numerical features
                        num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
                        
                        all_feature_names = list(num_features) + list(cat_features)
                        
                        coef_df = pd.DataFrame({
                            'feature': all_feature_names,
                            'coefficient': coef,
                            'odds_ratio': np.exp(coef),
                            'abs_coefficient': np.abs(coef)
                        }).sort_values('abs_coefficient', ascending=False)
                        
                        # Add interpretation
                        coef_df['interpretation'] = coef_df['coefficient'].apply(
                            lambda x: 'Increases churn risk' if x > 0 else 'Decreases churn risk'
                        )
                        
                        return coef_df
        
        return None
    
    def get_feature_importance_rf(self) -> Optional[pd.DataFrame]:
        """
        Extract Random Forest feature importance.
        
        Returns:
            DataFrame with feature importance scores
        """
        if hasattr(self.model, 'named_steps'):
            if 'model' in self.model.named_steps:
                rf_model = self.model.named_steps['model']
                
                if hasattr(rf_model, 'feature_importances_'):
                    importance = rf_model.feature_importances_
                    
                    # Get feature names after preprocessing
                    if hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
                        preprocessor = self.model.named_steps['preprocessor']
                        
                        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
                        cat_features = ohe.get_feature_names_out()
                        
                        num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
                        
                        all_feature_names = list(num_features) + list(cat_features)
                        
                        importance_df = pd.DataFrame({
                            'feature': all_feature_names,
                            'importance': importance
                        }).sort_values('importance', ascending=False)
                        
                        return importance_df
        
        return None


def explain_prediction(
    model: BaseEstimator,
    customer_data: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Generate a local explanation for a single customer prediction.
    
    This provides a rule-based explanation based on the customer's
    feature values, without requiring SHAP or other complex dependencies.
    
    Args:
        model: Trained model pipeline
        customer_data: DataFrame with single customer's features
        feature_names: List of feature names
        top_n: Number of top features to highlight
        
    Returns:
        Dictionary with prediction probability and explanation
    """
    # Get prediction
    proba = model.predict_proba(customer_data)[0, 1]
    prediction = int(proba >= 0.5)
    
    # Generate rule-based explanation
    reasons = []
    
    # Check key risk factors
    if 'Days_Since_Last_Transaction' in customer_data.columns:
        days_inactive = customer_data['Days_Since_Last_Transaction'].values[0]
        if days_inactive > 90:
            reasons.append(f"Inactive for {days_inactive} days (high risk)")
        elif days_inactive > 30:
            reasons.append(f"Inactive for {days_inactive} days (moderate risk)")
    
    if 'Number_of_Complaints' in customer_data.columns:
        complaints = customer_data['Number_of_Complaints'].values[0]
        if complaints >= 2:
            reasons.append(f"{complaints} complaints (elevated risk)")
        elif complaints == 1:
            reasons.append("1 complaint (moderate risk)")
    
    if 'Number_of_Products' in customer_data.columns:
        products = customer_data['Number_of_Products'].values[0]
        if products == 1:
            reasons.append("Single product (higher churn risk)")
        elif products >= 4:
            reasons.append(f"{products} products (lower churn risk)")
    
    if 'Digital_Engagement_Score' in customer_data.columns:
        digital_score = customer_data['Digital_Engagement_Score'].values[0]
        if digital_score < 40:
            reasons.append(f"Low digital engagement ({digital_score:.1f})")
        elif digital_score > 70:
            reasons.append(f"High digital engagement ({digital_score:.1f})")
    
    if 'Account_Tenure_Months' in customer_data.columns:
        tenure = customer_data['Account_Tenure_Months'].values[0]
        if tenure < 12:
            reasons.append(f"New customer ({tenure} months)")
        elif tenure > 60:
            reasons.append(f"Long-term customer ({tenure} months)")
    
    return {
        'churn_probability': float(proba),
        'predicted_class': 'Churn' if prediction == 1 else 'Retained',
        'risk_band': 'High Risk' if proba >= 0.66 else 'Medium Risk' if proba >= 0.33 else 'Low Risk',
        'reasons': reasons[:top_n]
    }


def get_shap_explanations(
    model: BaseEstimator,
    X: pd.DataFrame,
    max_samples: int = 100
) -> Optional[Dict[str, Any]]:
    """
    Generate SHAP explanations (optional dependency).
    
    This function only works if shap is installed. It provides
    local explanations for individual predictions and global
    feature importance.
    
    Args:
        model: Trained model pipeline
        X: Feature matrix
        max_samples: Maximum number of samples for explanation
        
    Returns:
        Dictionary with SHAP values or None if shap not available
    """
    try:
        import shap
    except ImportError:
        return None
    
    # Sample data if too large
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X
    
    # Create explainer based on model type
    if hasattr(model, 'predict_proba'):
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        
        return {
            'shap_values': shap_values,
            'feature_importance': np.abs(shap_values.values).mean(axis=0),
            'base_value': shap_values.base_values
        }
    
    return None


def generate_explanation_report(
    model: BaseEstainer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: List[str],
    model_type: str = "logistic"
) -> str:
    """
    Generate a comprehensive explainability report.
    
    Args:
        model: Trained model pipeline
        X_train: Training features
        y_train: Training target
        feature_names: Feature names
        model_type: Type of model ('logistic' or 'random_forest')
        
    Returns:
        String report with explainability findings
    """
    explainer = ModelExplainer(model, feature_names)
    
    report = ["=" * 60]
    report.append("MODEL EXPLAINABILITY REPORT")
    report.append("=" * 60)
    report.append(f"Model Type: {model_type}")
    report.append("")
    
    # Permutation importance
    report.append("PERMUTATION IMPORTANCE (Test Set)")
    report.append("-" * 60)
    perm_importance = explainer.get_permutation_importance(X_train, y_train)
    if perm_importance is not None:
        report.append(perm_importance.head(10).to_string(index=False))
    report.append("")
    
    # Model-specific importance
    if model_type == "logistic":
        report.append("LOGISTIC REGRESSION COEFFICIENTS")
        report.append("-" * 60)
        coef_df = explainer.get_logistic_coefficients()
        if coef_df is not None:
            report.append("Top 10 features by absolute coefficient:")
            report.append(coef_df.head(10)[['feature', 'coefficient', 'odds_ratio', 'interpretation']].to_string(index=False))
            report.append("")
            report.append("Note: Coefficients can be unstable with correlated features.")
            report.append("Digital engagement flags (mobile/internet banking) are correlated,")
            report.append("which can cause coefficient sign inversion - this is a known")
            report.append("multicollinearity artifact, not necessarily a causal signal.")
    elif model_type == "random_forest":
        report.append("RANDOM FOREST FEATURE IMPORTANCE")
        report.append("-" * 60)
        rf_importance = explainer.get_feature_importance_rf()
        if rf_importance is not None:
            report.append(rf_importance.head(10).to_string(index=False))
    
    report.append("")
    report.append("=" * 60)
    report.append("IMPORTANT CAVEATS")
    report.append("=" * 60)
    report.append("- Feature importance shows association, not causation.")
    report.append("- High importance does not mean changing the feature will reduce churn.")
    report.append("- Correlated features can have unstable coefficients (logistic regression).")
    report.append("- These explanations are based on synthetic data, not real customers.")
    report.append("- Any retention intervention should be tested with a controlled experiment.")
    report.append("=" * 60)
    
    return "\n".join(report)
