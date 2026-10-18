# Reproducibility Guide

This document describes how to reproduce the results in this project.

## Environment Setup

### Python Version
- Python 3.10 or higher recommended

### Dependencies
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning
- joblib: Model serialization
- streamlit: Prediction interface
- pytest: Testing
- shap: Model explainability (optional)

## Data Generation

The synthetic dataset is generated using `scripts/generate_dataset.py`:
```bash
python scripts/generate_dataset.py
```

This creates:
- `data/raw/churn_customers_synthetic.csv` - Raw synthetic data with injected quality issues

**Random Seed:** The generator uses a fixed random seed (42) for reproducibility.

## Model Training

Train the model using the modular pipeline:
```bash
python scripts/train_pipeline.py
```

This script:
1. Loads raw data
2. Validates data quality
3. Cleans data (removes duplicates, fixes outliers, winsorizes income)
4. Performs feature engineering (leakage-safe)
5. Splits data (75/25 stratified train/test)
6. Compares models (baseline, logistic regression, random forest)
7. Trains selected model (logistic regression)
8. Evaluates on test set
9. Performs threshold and capacity analysis
10. Runs explainability analysis
11. Generates retention analysis
12. Saves artifacts

### Artifacts Generated
- `models/logistic_regression_pipeline.joblib` - Trained model with metadata
- `data/processed/churn_predictions.csv` - Test set predictions with risk bands
- `data/processed/churn_customers_clean.csv` - Cleaned dataset

### Random Seed
The training script uses `random_state=42` by default for reproducibility.

## Model Performance (Test Set)

When run with the default random seed, the model achieves:

| Metric | Value |
|--------|-------|
| Accuracy | 0.6777 |
| Precision | 0.2271 |
| Recall | 0.6529 |
| F1 Score | 0.3370 |
| ROC-AUC | 0.7417 |
| Average Precision | 0.3534 |
| Log Loss | 0.5949 |
| Brier Score | 0.2055 |

### Baseline Comparison
The baseline (always predicts "no churn") achieves:
- Accuracy: 0.8745
- Recall: 0.0000
- ROC-AUC: 0.5000

Our model achieves 65.3% recall, identifying 65% of churned customers.

### Risk Band Calibration
- Low Risk: 3.9% observed churn rate
- Medium Risk: 12.1% observed churn rate
- High Risk: 33.1% observed churn rate

### Permutation Importance (Top 5)
1. Days_Since_Last_Transaction: 7.4%
2. Account_Tenure_Months: 5.3%
3. Number_of_Complaints: 4.1%
4. Number_of_Products: 2.9%
5. Average_Monthly_Deposit: 1.0%

## Running Tests

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Prediction Interface

Launch the Streamlit prediction interface:
```bash
streamlit run app.py
```

This provides:
- Single customer prediction
- Batch prediction from CSV upload
- Model information display

## Power BI Dashboard

See `dashboard/powerbi_dashboard_guide.md` for the Power BI specification.

## Important Notes

### Data Leakage Prevention
- Feature engineering uses training-set-only statistics
- Normalization parameters are fit on training data only
- No information from test set is used during training

### Synthetic Data
- All data is synthetic for educational purposes
- Nyanja Commercial Bank is a fictional institution
- Do not use this model for real banking decisions

### Reproducibility Checklist
- [ ] Use Python 3.10+
- [ ] Install all dependencies from requirements.txt
- [ ] Run data generation with default seed (42)
- [ ] Run training script with default random_state (42)
- [ ] Verify model metrics match documented values
- [ ] Run tests to verify code correctness

## Troubleshooting

### Import Errors
Ensure you're running from the project root directory:
```bash
cd banking-customer-churn-analysis
```

### Model Not Found
Run the training script first to generate the model:
```bash
python scripts/train_pipeline.py
```

### Test Failures
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Streamlit Errors
Ensure the model exists before launching the app:
```bash
python scripts/train_pipeline.py
streamlit run app.py
```
