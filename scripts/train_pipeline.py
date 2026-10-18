"""
Reproducible training script for banking customer churn prediction.

This script orchestrates the complete modeling workflow:
1. Load and validate data
2. Clean data (handle duplicates, outliers, invalid values)
3. Engineer features with leakage prevention
4. Split data (stratified train/test)
5. Train models with cross-validation
6. Evaluate on held-out test set
7. Select threshold
8. Save pipeline and results

Usage:
    python scripts/train_pipeline.py
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_validation import DataValidator, validate_dataset
from src.feature_engineering import FeatureEngineer, get_feature_definitions
from src.modeling import ChurnModelPipeline, ModelEvaluator, compare_models
from src.explainability import ModelExplainer, generate_explanation_report
from src.retention import RetentionRecommender, generate_retention_report


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """Load raw synthetic dataset."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset following the documented methodology.
    
    This function:
    - Drops exact duplicates
    - Standardizes categorical labels
    - Treats implausible ages as missing
    - Treats negative transaction counts as missing
    - Winsorizes extreme income at 99.5th percentile
    - Leaves missing values for pipeline imputation
    """
    print("\n=== Data Cleaning ===")
    df = df.copy()
    
    # 1. Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} duplicate rows -> {len(df):,} remaining")
    
    # 2. Standardize Customer_Segment labels
    segment_fix_map = {
        "retail": "Retail", "S.M.E": "SME", "PREMIUM": "Premium",
        "Salary account": "Salary Account", "Student / Youth": "Student/Youth",
    }
    df["Customer_Segment"] = df["Customer_Segment"].replace(segment_fix_map)
    print("Standardized Customer_Segment labels")
    
    # 3. Treat implausible ages as missing
    invalid_age_mask = (df["Age"] < 18) | (df["Age"] > 90)
    n_invalid_age = invalid_age_mask.sum()
    df.loc[invalid_age_mask, "Age"] = np.nan
    print(f"Treated {n_invalid_age} implausible ages as missing")
    
    # 4. Treat negative transaction counts as missing
    invalid_txn_mask = df["Average_Monthly_Transactions"] < 0
    n_invalid_txn = invalid_txn_mask.sum()
    df.loc[invalid_txn_mask, "Average_Monthly_Transactions"] = np.nan
    print(f"Treated {n_invalid_txn} negative transaction counts as missing")
    
    # 5. Winsorize extreme income at 99.5th percentile
    income_cap = df["Monthly_Income"].quantile(0.995)
    n_capped = (df["Monthly_Income"] > income_cap).sum()
    df["Monthly_Income"] = df["Monthly_Income"].clip(upper=income_cap)
    print(f"Winsorized income at {income_cap:,.0f} ({n_capped} rows affected)")
    
    # Report remaining missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\nRemaining missing values (will be imputed in pipeline):")
        for col, count in missing_cols.items():
            print(f"  {col}: {count}")
    
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature columns and target.
    
    Returns:
        tuple: (X, y, numerical_cols, categorical_cols)
    """
    # Define feature columns
    numerical_cols = [
        "Age", "Monthly_Income", "Account_Tenure_Months", "Number_of_Products",
        "Has_Credit_Card", "Has_Mobile_Banking", "Has_Internet_Banking",
        "Average_Monthly_Balance", "Average_Monthly_Transactions",
        "Total_Transaction_Value_Last_6_Months", "Number_of_Complaints",
        "Days_Since_Last_Transaction", "Average_Monthly_Deposit", "Average_Monthly_Withdrawal",
        "Has_Active_Loan", "Digital_Engagement_Score", "Customer_Service_Interactions",
    ]
    
    categorical_cols = [
        "Gender", "Marital_Status", "Education_Level", "Employment_Status",
        "Customer_Segment", "Branch_Location", "Loan_Status",
    ]
    
    # Exclude Customer_ID from features
    feature_cols = numerical_cols + categorical_cols
    
    X = df[feature_cols].copy()
    y = df["Churn"].copy()
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, numerical_cols, categorical_cols


def main():
    """Main training workflow."""
    parser = argparse.ArgumentParser(description="Train churn prediction pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/churn_customers_synthetic.csv",
        help="Path to raw data file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Proportion of data for test set"
    )
    
    args = parser.parse_args()
    
    np.random.seed(args.random_state)
    
    print("=" * 60)
    print("BANKING CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = load_raw_data(data_path)
    
    # 2. Validate data
    print("\n=== Data Validation ===")
    validator = DataValidator(df)
    result = validator.validate_all()
    print(validator.generate_report())
    
    if not result.passed:
        print("WARNING: Validation failed. Proceeding with caution.")
    
    # 3. Clean data
    df_clean = clean_data(df)
    
    # 4. Prepare features
    X, y, numerical_cols, categorical_cols = prepare_features(df_clean)
    
    # 5. Train/test split (stratified)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train):,} rows ({y_train.mean():.2%} churn rate)")
    print(f"Test set:  {len(X_test):,} rows ({y_test.mean():.2%} churn rate)")
    
    # 6. Compare models
    print("\n=== Model Comparison ===")
    comparison_df = compare_models(
        X_train, y_train, X_test, y_test,
        numerical_cols, categorical_cols,
        random_state=args.random_state
    )
    print(comparison_df.round(4))
    
    # 7. Train selected model (Logistic Regression for interpretability)
    print("\n=== Training Selected Model (Logistic Regression) ===")
    selected_pipeline = ChurnModelPipeline(
        model_type="logistic",
        random_state=args.random_state,
        class_weight="balanced"
    )
    selected_pipeline.fit(X_train, y_train, numerical_cols, categorical_cols)
    
    # 8. Evaluate on test set
    print("\n=== Test Set Evaluation ===")
    evaluator = ModelEvaluator()
    y_pred = selected_pipeline.predict(X_test)
    y_proba = selected_pipeline.predict_proba(X_test)
    
    test_metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)
    print("\nTest set metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 9. Threshold analysis
    print("\n=== Threshold Analysis ===")
    thresholds = np.arange(0.30, 0.71, 0.05)
    threshold_df = evaluator.threshold_analysis(y_test, y_proba, thresholds)
    print(threshold_df.round(4))
    
    # 10. Capacity-constrained analysis
    print("\n=== Capacity-Constrained Analysis ===")
    capacities = np.array([100, 500, 1000, 1500, 2000])
    capacity_df = evaluator.capacity_constrained_analysis(y_test, y_proba, capacities)
    print(capacity_df.round(4))
    
    # 11. Calibration analysis
    print("\n=== Calibration Analysis ===")
    calibration_results = evaluator.calibration_analysis(y_test, y_proba)
    print(f"Expected Calibration Error (ECE): {calibration_results['ece']:.4f}")
    
    # 12. Explainability analysis
    print("\n=== Explainability Analysis ===")
    explainer = ModelExplainer(selected_pipeline.pipeline, numerical_cols + categorical_cols)
    perm_importance = explainer.get_permutation_importance(X_test, y_test, n_repeats=5)
    print("Top 10 features by permutation importance:")
    print(perm_importance.head(10).to_string(index=False))
    
    # 10. Save pipeline
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "logistic_regression_pipeline.joblib"
    
    metadata = {
        "test_metrics": test_metrics,
        "threshold_analysis": threshold_df.to_dict(),
        "capacity_analysis": capacity_df.to_dict(),
        "calibration_ece": float(calibration_results['ece']),
        "feature_definitions": get_feature_definitions(),
        "data_info": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
        },
        "random_state": args.random_state,
    }
    
    selected_pipeline.save(model_path, metadata)
    print(f"\nModel saved to {model_path}")
    
    # 11. Save predictions
    predictions_df = pd.DataFrame({
        "Customer_ID": df_clean.loc[X_test.index, "Customer_ID"],
        "Actual_Churn": y_test.values,
        "Churn_Probability": y_proba[:, 1],
        "Predicted_Churn_0.50": y_pred,
    })
    
    # Add risk bands
    predictions_df["Risk_Band"] = pd.cut(
        predictions_df["Churn_Probability"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )
    
    predictions_path = output_dir.parent / "data" / "processed" / "churn_predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    # 13. Retention analysis
    print("\n=== Retention Analysis ===")
    recommender = RetentionRecommender()
    retention_report = generate_retention_report(predictions_df)
    print(retention_report)
    
    # 12. Save cleaned data
    cleaned_path = output_dir.parent / "data" / "processed" / "churn_customers_clean.csv"
    df_clean.to_csv(cleaned_path, index=False)
    print(f"Cleaned data saved to {cleaned_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
