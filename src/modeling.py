"""
Model training and evaluation for banking customer churn prediction.

This module provides functions for building reproducible modeling pipelines
with proper train/test separation, cross-validation, and evaluation metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
import joblib
import json


class ChurnModelPipeline:
    """
    Reproducible churn modeling pipeline with leakage prevention.
    
    This class encapsulates the full modeling workflow:
    1. Data preprocessing (imputation, scaling, encoding)
    2. Feature engineering (with training-only statistics)
    3. Model training
    4. Evaluation
    """
    
    def __init__(
        self,
        model_type: str = "logistic",
        random_state: int = 42,
        class_weight: str = "balanced"
    ):
        """
        Initialize the modeling pipeline.
        
        Args:
            model_type: Type of model ('logistic', 'random_forest', or 'baseline')
            random_state: Random seed for reproducibility
            class_weight: Class weighting strategy
        """
        self.model_type = model_type
        self.random_state = random_state
        self.class_weight = class_weight
        self.pipeline = None
        self.feature_names_in_ = None
        self.is_fitted_ = False
        
    def _get_preprocessor(self, numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
        """
        Create preprocessing transformer.
        
        Args:
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            
        Returns:
            ColumnTransformer with imputation, scaling, and encoding
        """
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def _get_model(self) -> BaseEstimator:
        """Get the model instance based on model_type."""
        if self.model_type == "logistic":
            return LogisticRegression(
                random_state=self.random_state,
                class_weight=self.class_weight,
                max_iter=1000,
                solver='lbfgs'
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                random_state=self.random_state,
                class_weight=self.class_weight,
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                n_jobs=-1
            )
        elif self.model_type == "baseline":
            # Baseline: always predict majority class (0 = retained)
            class MajorityBaseline(BaseEstimator, ClassifierMixin):
                def __init__(self):
                    self.majority_class_ = 0
                
                def fit(self, X, y):
                    self.majority_class_ = int(y.mode()[0])
                    return self
                
                def predict(self, X):
                    return np.array([self.majority_class_] * len(X))
                
                def predict_proba(self, X):
                    if self.majority_class_ == 0:
                        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])
                    else:
                        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])
            
            return MajorityBaseline()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def build_pipeline(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str],
        use_feature_engineering: bool = True
    ) -> Pipeline:
        """
        Build the complete modeling pipeline.
        
        Args:
            numerical_cols: List of numerical feature columns
            categorical_cols: List of categorical feature columns
            use_feature_engineering: Whether to include feature engineering step
            
        Returns:
            Scikit-learn Pipeline
        """
        from src.feature_engineering import FeatureEngineer
        
        steps = []
        
        if use_feature_engineering:
            steps.append(('feature_engineer', FeatureEngineer()))
        
        steps.append(('preprocessor', self._get_preprocessor(numerical_cols, categorical_cols)))
        steps.append(('model', self._get_model()))
        
        self.pipeline = Pipeline(steps)
        return self.pipeline
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        numerical_cols: List[str],
        categorical_cols: List[str],
        use_feature_engineering: bool = True
    ):
        """
        Fit the pipeline on training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            numerical_cols: Numerical column names
            categorical_cols: Categorical column names
            use_feature_engineering: Whether to use feature engineering
            
        Returns:
            self
        """
        self.feature_names_in_ = list(X_train.columns)
        self.build_pipeline(numerical_cols, categorical_cols, use_feature_engineering)
        self.pipeline.fit(X_train, y_train)
        self.is_fitted_ = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted before prediction")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted before prediction")
        return self.pipeline.predict_proba(X)
    
    def save(self, path: Path, metadata: Optional[Dict] = None):
        """
        Save the pipeline and metadata to disk.
        
        Args:
            path: Path to save the pipeline
            metadata: Optional metadata dictionary to save
        """
        if not self.is_fitted_:
            raise ValueError("Cannot save an unfitted pipeline")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'pipeline': self.pipeline,
            'model_type': self.model_type,
            'random_state': self.random_state,
            'class_weight': self.class_weight,
            'feature_names_in_': self.feature_names_in_,
            'metadata': metadata or {}
        }
        
        joblib.dump(save_dict, path)
    
    @classmethod
    def load(cls, path: Path) -> 'ChurnModelPipeline':
        """
        Load a pipeline from disk.
        
        Args:
            path: Path to the saved pipeline
            
        Returns:
            ChurnModelPipeline instance
        """
        save_dict = joblib.load(path)
        
        instance = cls(
            model_type=save_dict['model_type'],
            random_state=save_dict['random_state'],
            class_weight=save_dict['class_weight']
        )
        instance.pipeline = save_dict['pipeline']
        instance.feature_names_in_ = save_dict['feature_names_in_']
        instance.is_fitted_ = True
        
        return instance


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics."""
    
    def __init__(self, positive_class: int = 1):
        """
        Initialize evaluator.
        
        Args:
            positive_class: Which class is considered positive (churn)
        """
        self.positive_class = positive_class
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for AUC metrics)
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label=self.positive_class, zero_division=0),
            'recall': recall_score(y_true, y_pred, pos_label=self.positive_class, zero_division=0),
            'f1': f1_score(y_true, y_pred, pos_label=self.positive_class, zero_division=0),
            'specificity': recall_score(y_true, y_pred, pos_label=1-self.positive_class, zero_division=0),
            'balanced_accuracy': (recall_score(y_true, y_pred, pos_label=self.positive_class, zero_division=0) + 
                                 recall_score(y_true, y_pred, pos_label=1-self.positive_class, zero_division=0)) / 2
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            metrics['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
            metrics['log_loss'] = log_loss(y_true, y_proba)
            metrics['brier_score'] = brier_score_loss(y_true, y_proba[:, 1])
        
        return metrics
    
    def confusion_matrix_at_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Calculate confusion matrix at a specific threshold.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            threshold: Decision threshold
            
        Returns:
            Tuple of (confusion matrix array, dict with TP, FP, TN, FN counts)
        """
        y_pred = (y_proba[:, 1] >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        
        return cm, {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
    
    def threshold_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: np.ndarray
    ) -> pd.DataFrame:
        """
        Analyze performance across multiple thresholds.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            thresholds: Array of threshold values to test
            
        Returns:
            DataFrame with metrics at each threshold
        """
        results = []
        
        for thresh in thresholds:
            y_pred = (y_proba[:, 1] >= thresh).astype(int)
            cm, counts = self.confusion_matrix_at_threshold(y_true, y_proba, thresh)
            
            metrics = self.calculate_metrics(y_true, y_pred)
            
            results.append({
                'threshold': thresh,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'true_positives': counts['true_positives'],
                'false_positives': counts['false_positives'],
                'false_negatives': counts['false_negatives'],
                'customers_flagged': counts['true_positives'] + counts['false_positives'],
                'share_flagged': (counts['true_positives'] + counts['false_positives']) / len(y_true)
            })
        
        return pd.DataFrame(results)
    
    def capacity_constrained_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        capacities: np.ndarray
    ) -> pd.DataFrame:
        """
        Analyze performance under capacity constraints (top K customers).
        
        Args:
            y_true: True labels (can be pandas Series or numpy array)
            y_proba: Predicted probabilities
            capacities: Array of K values (number of customers to review)
            
        Returns:
            DataFrame with metrics at each capacity level
        """
        results = []
        
        # Convert to numpy arrays if needed
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_true, 'to_numpy'):
            y_true = y_true.to_numpy()
        
        # Sort by probability descending
        sorted_indices = np.argsort(y_proba[:, 1])[::-1]
        
        for capacity in capacities:
            capacity = min(capacity, len(y_true))
            top_k_indices = sorted_indices[:capacity]
            
            y_pred = np.zeros(len(y_true), dtype=int)
            y_pred[top_k_indices] = 1
            
            # Recalculate counts for capacity-based selection
            tp = ((y_true[top_k_indices] == 1) & (y_pred[top_k_indices] == 1)).sum()
            fp = ((y_true[top_k_indices] == 0) & (y_pred[top_k_indices] == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            
            results.append({
                'capacity': capacity,
                'share_of_customers': capacity / len(y_true),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'min_probability': float(y_proba[top_k_indices[-1], 1]) if capacity > 0 else 0.0
            })
        
        return pd.DataFrame(results)
    
    def calibration_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze calibration of probability predictions.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics and curve data
        """
        prob_true, prob_pred = calibration_curve(y_true, y_proba[:, 1], n_bins=n_bins, strategy='uniform')
        
        # Expected calibration error
        ece = 0.0
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_width = 1.0 / n_bins
        
        for i in range(n_bins):
            mask = (y_proba[:, 1] > bin_edges[i]) & (y_proba[:, 1] <= bin_edges[i + 1])
            if mask.sum() > 0:
                ece += (mask.sum() / len(y_true)) * abs(prob_true[i] - prob_pred[i])
        
        return {
            'ece': ece,
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'n_bins': n_bins
        }
    
    def cross_validate(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'roc_auc'
    ) -> Dict[str, Any]:
        """
        Perform stratified cross-validation.
        
        Args:
            pipeline: Fitted or unfitted pipeline
            X: Features
            y: Target
            cv: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV scores and statistics
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring=scoring)
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max(),
            'scoring': scoring
        }


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numerical_cols: List[str],
    categorical_cols: List[str],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare multiple models using the same evaluation protocol.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        numerical_cols: Numerical column names
        categorical_cols: Categorical column names
        random_state: Random seed
        
    Returns:
        DataFrame with comparison results
    """
    models = ['baseline', 'logistic', 'random_forest']
    results = []
    
    evaluator = ModelEvaluator()
    
    for model_type in models:
        print(f"Training {model_type}...")
        
        # Baseline doesn't need feature engineering or preprocessing
        use_fe = model_type != 'baseline'
        class_weight = 'balanced' if model_type != 'baseline' else None
        
        pipeline = ChurnModelPipeline(
            model_type=model_type,
            random_state=random_state,
            class_weight=class_weight
        )
        
        pipeline.fit(X_train, y_train, numerical_cols, categorical_cols, use_feature_engineering=use_fe)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)
        metrics['model'] = model_type
        
        results.append(metrics)
    
    return pd.DataFrame(results).set_index('model')
