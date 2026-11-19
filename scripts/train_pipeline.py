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
    - Drops repeated customer IDs (including exact duplicate rows)
    - Standardizes categorical labels
    - Treats implausible ages as missing
    - Treats negative transaction counts as missing
    - Winsorizes extreme income at 99.5th percentile
    - Leaves missing values for pipeline imputation
    """
    print("\n=== Data Cleaning ===")
    df = df.copy()
    
    # 1. Enforce one record per customer. Some injected duplicate rows are
    # subsequently altered by other quality injections, so exact-row deduping
    # alone would leave conflicting records for the same customer ID.
    before = len(df)
    df = df.drop_duplicates(subset=["Customer_ID"], keep="first")
    print(f"Dropped {before - len(df):,} duplicate customer records -> {len(df):,} remaining")
    
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
    
    # 6. Hold back part of the training partition for candidate comparison and threshold scenarios.
    # The final test partition is reserved exclusively for one-time reporting of the selected model.
    from sklearn.model_selection import train_test_split
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train, y_train, test_size=0.20, random_state=args.random_state, stratify=y_train
    )

    # 7. Compare candidate models on validation data, never on the final test set.
    print("\n=== Validation-Set Model Comparison ===")
    comparison_df = compare_models(
        X_fit, y_fit, X_validation, y_validation,
        numerical_cols, categorical_cols,
        random_state=args.random_state
    )
    print(comparison_df.round(4))

    # Fit a validation-only model for threshold scenario analysis.
    validation_pipeline = ChurnModelPipeline(
        model_type="logistic", random_state=args.random_state, class_weight="balanced"
    )
    validation_pipeline.fit(X_fit, y_fit, numerical_cols, categorical_cols)
    validation_proba = validation_pipeline.predict_proba(X_validation)

    # 8. Refit the production candidate on the complete training partition.
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
    
    # 9. Threshold scenarios are evaluated on validation data, not the final test set.
    print("\n=== Validation-Set Threshold Analysis ===")
    thresholds = np.arange(0.10, 0.91, 0.05)
    threshold_df = evaluator.threshold_analysis(y_validation, validation_proba, thresholds)
    print(threshold_df.round(4))

    # 10. Cross-validation is restricted to training data; preprocessing is inside the pipeline.
    print("\n=== Training-Only Stratified Cross-Validation ===")
    cv_roc_auc = evaluator.cross_validate(
        selected_pipeline.pipeline, X_train, y_train, cv=5, scoring="roc_auc"
    )
    cv_average_precision = evaluator.cross_validate(
        selected_pipeline.pipeline, X_train, y_train, cv=5, scoring="average_precision"
    )
    print(f"ROC-AUC: {cv_roc_auc['mean']:.4f} +/- {cv_roc_auc['std']:.4f}")
    print(f"Average precision: {cv_average_precision['mean']:.4f} +/- {cv_average_precision['std']:.4f}")

    # 11. Final-test diagnostics at the unchanged 0.50 operating threshold.
    test_confusion, confusion_counts = evaluator.confusion_matrix_at_threshold(
        y_test, y_proba, threshold=0.50
    )
    from sklearn.metrics import precision_recall_curve
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_proba[:, 1])
    subgroup_metrics = {}
    for group_column in ["Branch_Location", "Customer_Segment"]:
        subgroup_metrics[group_column] = evaluator.subgroup_performance(
            y_test, y_pred, X_test[group_column], group_name=group_column, min_group_size=50
        ).to_dict(orient="records")
    
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
        "model_comparison": comparison_df.reset_index().to_dict(orient="records"),
        "test_confusion_matrix": test_confusion.tolist(),
        "test_confusion_counts": confusion_counts,
        "pr_curve": {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
        "threshold_analysis": threshold_df.to_dict(),
        "threshold_analysis_source": "Validation split drawn from training data; final test set not used for threshold scenarios.",
        "cross_validation": {
            "roc_auc": {key: (value.tolist() if isinstance(value, np.ndarray) else float(value) if isinstance(value, (np.floating, float)) else value) for key, value in cv_roc_auc.items()},
            "average_precision": {key: (value.tolist() if isinstance(value, np.ndarray) else float(value) if isinstance(value, (np.floating, float)) else value) for key, value in cv_average_precision.items()},
        },
        "subgroup_metrics": subgroup_metrics,
        "capacity_analysis": capacity_df.to_dict(),
        "calibration_ece": float(calibration_results['ece']),
        "calibration_curve": {
            "prob_true": calibration_results["prob_true"].tolist(),
            "prob_pred": calibration_results["prob_pred"].tolist(),
            "bin_count": calibration_results["bin_count"].tolist(),
            "bin_edges": calibration_results["bin_edges"].tolist(),
        },
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
    
    # Keep the held-out test predictions separate for honest model evaluation.
    risk_recommender = RetentionRecommender()
    predictions_df["Risk_Band"] = predictions_df["Churn_Probability"].apply(
        risk_recommender.assign_risk_band
    )

    processed_dir = output_dir.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = processed_dir / "churn_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Held-out test predictions saved to {predictions_path}")

    # Score the complete cleaned customer population for portfolio monitoring
    # and capacity-constrained retention prioritisation. These are in-sample
    # portfolio scores, not an independent evaluation of model performance.
    portfolio_probabilities = selected_pipeline.predict_proba(X)[:, 1]
    portfolio_scores = pd.DataFrame({
        "Customer_ID": df_clean["Customer_ID"].values,
        "Churn_Probability": portfolio_probabilities,
        "Predicted_Churn_0.50": (portfolio_probabilities >= 0.50).astype(int),
    })
    portfolio_scores["Risk_Band"] = portfolio_scores["Churn_Probability"].apply(
        risk_recommender.assign_risk_band
    )
    portfolio_path = processed_dir / "churn_customer_scores.csv"
    portfolio_scores.to_csv(portfolio_path, index=False)
    print(f"Full-population customer scores saved to {portfolio_path}")
    
    # 13. Retention analysis
    print("\n=== Retention Analysis ===")
    recommender = risk_recommender
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
