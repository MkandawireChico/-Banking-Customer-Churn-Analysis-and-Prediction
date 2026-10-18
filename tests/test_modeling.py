"""
Tests for modeling module.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.modeling import ChurnModelPipeline, ModelEvaluator


@pytest.fixture
def sample_data():
    """Create sample data for modeling tests."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Age": np.random.randint(18, 75, n),
        "Monthly_Income": np.random.lognormal(13, 0.5, n),
        "Account_Tenure_Months": np.random.randint(1, 240, n),
        "Number_of_Products": np.random.randint(1, 7, n),
        "Has_Credit_Card": np.random.binomial(1, 0.3, n),
        "Has_Mobile_Banking": np.random.binomial(1, 0.7, n),
        "Has_Internet_Banking": np.random.binomial(1, 0.5, n),
        "Average_Monthly_Balance": np.random.lognormal(13, 0.5, n),
        "Average_Monthly_Transactions": np.random.poisson(8, n),
        "Total_Transaction_Value_Last_6_Months": np.random.lognormal(14, 0.5, n),
        "Number_of_Complaints": np.random.poisson(0.5, n),
        "Days_Since_Last_Transaction": np.random.exponential(30, n),
        "Average_Monthly_Deposit": np.random.lognormal(13, 0.5, n),
        "Average_Monthly_Withdrawal": np.random.lognormal(13, 0.5, n),
        "Has_Active_Loan": np.random.binomial(1, 0.3, n),
        "Digital_Engagement_Score": np.random.uniform(0, 100, n),
        "Customer_Service_Interactions": np.random.poisson(0.5, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Marital_Status": np.random.choice(["Single", "Married", "Divorced", "Widowed"], n),
        "Education_Level": np.random.choice(["Primary", "Secondary", "Tertiary/College", "University"], n),
        "Employment_Status": np.random.choice(["Employed - Private", "Employed - Government", "Business Owner", "Self-Employed", "Unemployed"], n),
        "Customer_Segment": np.random.choice(["Retail", "SME", "Premium", "Salary Account", "Student/Youth"], n),
        "Branch_Location": np.random.choice(["Blantyre", "Lilongwe", "Mzuzu", "Zomba", "Mangochi"], n),
        "Loan_Status": np.random.choice(["No Active Loan", "Active", "Active - Good Standing", "Active - Past Due"], n),
        "Churn": np.random.binomial(1, 0.15, n),
    })


def test_churn_model_pipeline_init():
    """Test ChurnModelPipeline initialization."""
    pipeline = ChurnModelPipeline(model_type="logistic", random_state=42)
    assert pipeline.model_type == "logistic"
    assert pipeline.random_state == 42
    assert pipeline.is_fitted_ == False


def test_churn_model_pipeline_invalid_model_type(sample_data):
    """Test ChurnModelPipeline with invalid model type."""
    numerical_cols = ["Age", "Monthly_Income"]
    categorical_cols = ["Gender"]
    
    X = sample_data[numerical_cols + categorical_cols]
    y = sample_data["Churn"]
    
    pipeline = ChurnModelPipeline(model_type="invalid_model", random_state=42)
    with pytest.raises(ValueError, match="Unknown model_type"):
        pipeline.fit(X, y, numerical_cols, categorical_cols)


def test_churn_model_pipeline_fit(sample_data):
    """Test ChurnModelPipeline fit."""
    numerical_cols = ["Age", "Monthly_Income", "Account_Tenure_Months", "Number_of_Products"]
    categorical_cols = ["Gender", "Customer_Segment"]
    
    X = sample_data[numerical_cols + categorical_cols]
    y = sample_data["Churn"]
    
    pipeline = ChurnModelPipeline(model_type="logistic", random_state=42)
    pipeline.fit(X, y, numerical_cols, categorical_cols)
    
    assert pipeline.is_fitted_ == True
    assert pipeline.pipeline is not None


def test_churn_model_pipeline_predict_without_fit(sample_data):
    """Test that predict raises error if not fitted."""
    numerical_cols = ["Age", "Monthly_Income"]
    categorical_cols = ["Gender"]
    
    X = sample_data[numerical_cols + categorical_cols]
    
    pipeline = ChurnModelPipeline(model_type="logistic", random_state=42)
    with pytest.raises(ValueError, match="must be fitted"):
        pipeline.predict(X)


def test_churn_model_pipeline_predict(sample_data):
    """Test ChurnModelPipeline predict."""
    numerical_cols = ["Age", "Monthly_Income", "Account_Tenure_Months", "Number_of_Products"]
    categorical_cols = ["Gender", "Customer_Segment"]
    
    X = sample_data[numerical_cols + categorical_cols]
    y = sample_data["Churn"]
    
    pipeline = ChurnModelPipeline(model_type="logistic", random_state=42)
    pipeline.fit(X, y, numerical_cols, categorical_cols)
    
    predictions = pipeline.predict(X)
    
    assert len(predictions) == len(X)
    assert all(p in [0, 1] for p in predictions)


def test_churn_model_pipeline_predict_proba(sample_data):
    """Test ChurnModelPipeline predict_proba."""
    numerical_cols = ["Age", "Monthly_Income", "Account_Tenure_Months", "Number_of_Products"]
    categorical_cols = ["Gender", "Customer_Segment"]
    
    X = sample_data[numerical_cols + categorical_cols]
    y = sample_data["Churn"]
    
    pipeline = ChurnModelPipeline(model_type="logistic", random_state=42)
    pipeline.fit(X, y, numerical_cols, categorical_cols)
    
    proba = pipeline.predict_proba(X)
    
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
    assert (proba >= 0).all() and (proba <= 1).all()  # Valid probability range


def test_churn_model_pipeline_random_forest(sample_data):
    """Test ChurnModelPipeline with Random Forest."""
    numerical_cols = ["Age", "Monthly_Income"]
    categorical_cols = ["Gender"]
    
    X = sample_data[numerical_cols + categorical_cols]
    y = sample_data["Churn"]
    
    pipeline = ChurnModelPipeline(model_type="random_forest", random_state=42)
    pipeline.fit(X, y, numerical_cols, categorical_cols)
    
    predictions = pipeline.predict(X)
    assert len(predictions) == len(X)


def test_churn_model_pipeline_baseline(sample_data):
    """Test ChurnModelPipeline with baseline model."""
    numerical_cols = ["Age", "Monthly_Income"]
    categorical_cols = ["Gender"]
    
    X = sample_data[numerical_cols + categorical_cols]
    y = sample_data["Churn"]
    
    pipeline = ChurnModelPipeline(model_type="baseline", random_state=42)
    pipeline.fit(X, y, numerical_cols, categorical_cols, use_feature_engineering=False)
    
    predictions = pipeline.predict(X)
    # Baseline should predict the same value for all
    assert len(set(predictions)) == 1


def test_model_evaluator_init():
    """Test ModelEvaluator initialization."""
    evaluator = ModelEvaluator(positive_class=1)
    assert evaluator.positive_class == 1


def test_model_evaluator_calculate_metrics():
    """Test ModelEvaluator calculate_metrics."""
    evaluator = ModelEvaluator()
    
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1


def test_model_evaluator_calculate_metrics_with_proba():
    """Test ModelEvaluator calculate_metrics with probabilities."""
    evaluator = ModelEvaluator()
    
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    y_proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.9, 0.1], [0.6, 0.4], [0.8, 0.2], [0.3, 0.7]])
    
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_proba)
    
    assert "roc_auc" in metrics
    assert "average_precision" in metrics
    assert "log_loss" in metrics
    assert "brier_score" in metrics


def test_model_evaluator_confusion_matrix_at_threshold():
    """Test ModelEvaluator confusion_matrix_at_threshold."""
    evaluator = ModelEvaluator()
    
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.9, 0.1], [0.6, 0.4], [0.8, 0.2], [0.3, 0.7]])
    
    cm, counts = evaluator.confusion_matrix_at_threshold(y_true, y_proba, threshold=0.5)
    
    assert "true_positives" in counts
    assert "false_positives" in counts
    assert "false_negatives" in counts
    assert "true_negatives" in counts
    assert counts["true_positives"] + counts["false_positives"] + counts["false_negatives"] + counts["true_negatives"] == len(y_true)


def test_model_evaluator_threshold_analysis():
    """Test ModelEvaluator threshold_analysis."""
    evaluator = ModelEvaluator()
    
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.9, 0.1], [0.6, 0.4], [0.8, 0.2], [0.3, 0.7]])
    thresholds = np.array([0.3, 0.5, 0.7])
    
    result = evaluator.threshold_analysis(y_true, y_proba, thresholds)
    
    assert len(result) == len(thresholds)
    assert "threshold" in result.columns
    assert "precision" in result.columns
    assert "recall" in result.columns
    assert "customers_flagged" in result.columns


def test_model_evaluator_capacity_constrained_analysis():
    """Test ModelEvaluator capacity_constrained_analysis."""
    evaluator = ModelEvaluator()
    
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.9, 0.1], [0.6, 0.4], [0.8, 0.2], 
                       [0.3, 0.7], [0.5, 0.5], [0.1, 0.9], [0.4, 0.6], [0.2, 0.8]])
    capacities = np.array([2, 5])
    
    result = evaluator.capacity_constrained_analysis(y_true, y_proba, capacities)
    
    assert len(result) == len(capacities)
    assert "capacity" in result.columns
    assert "precision" in result.columns
    assert "recall" in result.columns
    assert result.iloc[0]["capacity"] == 2
    assert result.iloc[1]["capacity"] == 5


def test_model_evaluator_calibration_analysis():
    """Test ModelEvaluator calibration_analysis."""
    evaluator = ModelEvaluator()
    
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.9, 0.1], [0.6, 0.4], [0.8, 0.2], 
                       [0.3, 0.7], [0.5, 0.5], [0.1, 0.9], [0.4, 0.6], [0.2, 0.8]])
    
    result = evaluator.calibration_analysis(y_true, y_proba, n_bins=5)
    
    assert "ece" in result
    assert "prob_true" in result
    assert "prob_pred" in result
    assert 0 <= result["ece"] <= 1
    assert len(result["prob_true"]) == 5
