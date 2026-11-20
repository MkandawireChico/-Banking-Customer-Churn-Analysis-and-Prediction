# Reproducibility and Model Operations

This guide describes how to regenerate the controlled customer cohort, train the churn model and review the resulting artefacts.

## Environment

Use Python 3.10, 3.11 or 3.12, then install the pinned project dependencies:

```bash
pip install -r requirements.txt
```

## 1. Generate the source cohort

```bash
python scripts/generate_dataset.py
```

The generator uses a fixed default seed and creates `data/raw/churn_customers_synthetic.csv`. Optional arguments can change the seed, population size or output path:

```bash
python scripts/generate_dataset.py --seed 42 --rows 20000
```

The generated records include controlled data-quality issues so the validation and cleaning steps can be demonstrated. They are synthetic and do not represent real bank customers.

## 2. Train and evaluate the model

```bash
python scripts/train_pipeline.py
```

The canonical pipeline:

1. Validates the raw data and reports quality issues.
2. Removes duplicate customer IDs, standardises categories and handles implausible values.
3. Leaves missing-value imputation inside the model pipeline.
4. Creates a stratified training and independent test split.
5. Uses a validation subset of the training partition for candidate comparison and threshold scenarios.
6. Retains Logistic Regression as the initial recall-oriented operating candidate.
7. Fits a sigmoid calibrator using five-fold cross-validation within the full training partition.
8. Evaluates the calibrated model once on the untouched test partition.
9. Calculates discrimination, classification, calibration, capacity and subgroup diagnostics.
10. Saves the model, metadata, test predictions, cleaned data and full-population scores.

The final test partition is not used for model selection, threshold selection or calibrator fitting.

## 3. Understand the generated artefacts

| Artefact | Purpose |
|---|---|
| `models/logistic_regression_pipeline.joblib` | Current calibrated pipeline and its metadata |
| `data/processed/churn_customers_clean.csv` | Cleaned customer profiles |
| `data/processed/churn_predictions.csv` | Held-out test predictions with actual outcomes |
| `data/processed/churn_customer_scores.csv` | Full-population scores for portfolio prioritisation |

**Important:** the full-population score file includes records from the model's training cohort. It is an operational portfolio-scoring output, not an independent evaluation sample. Report model performance from the held-out test predictions and saved model metadata.

The model metadata contains the test metrics, candidate comparison, confusion matrix, validation-only threshold analysis, cross-validation results, subgroup diagnostics, capacity analysis and calibration evidence. The dashboard and notebook read these saved results rather than relying on stale hard-coded metric values.

## 4. Review the notebook

Open `notebooks/customer_churn_analysis.ipynb` after generating the data and training artefacts. It provides:

- Data integrity and missingness checks.
- Portfolio-level and segment-level exploratory analysis.
- Behavioural churn patterns.
- A review of validation-only candidate comparison.
- Independent test metrics and confusion-matrix diagnostics.
- Risk-band outcome checks and calibration review.
- A clear separation between held-out evaluation and full-population scores.

The notebook is intentionally not a second, competing model-training implementation. The modular script in `scripts/train_pipeline.py` is the source of truth for model training and saved production artefacts.

## 5. Run the application and tests

```bash
pytest tests/ -v
streamlit run dashboard/streamlit_command_centre.py
```

The Streamlit app reads the processed customer and score exports plus the saved model bundle. The Campaign Studio and Case Management workspaces support CSV export/import handoffs; they do not currently persist records to a shared CRM database.

## 6. Controlled GitHub refresh

A manual workflow is available at:

[Refresh production model](https://github.com/MkandawireChico/-Banking-Customer-Churn-Analysis-and-Prediction/actions/workflows/refresh-production-model.yml)

It requires a deliberate confirmation before regenerating and committing model artefacts. It is not configured to retrain automatically on every push.

## Interpretation safeguards

- Accuracy alone is inadequate for an imbalanced churn problem.
- Precision, recall, F1, ROC-AUC and average precision describe different aspects of model performance.
- Brier score, log loss, calibration curves and ECE help assess probability quality.
- A high ECE should trigger caution: use scores as relative ranking signals rather than precise individual probabilities.
- Risk bands are prioritisation aids, not guarantees.
- Descriptive patterns in synthetic data are not causal evidence.
- Real deployment requires representative authorised data, privacy controls, fairness review, monitoring and human oversight.
