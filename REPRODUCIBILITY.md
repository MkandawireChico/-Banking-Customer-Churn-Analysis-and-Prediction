# Reproducibility Guide

This guide describes how to regenerate the customer dataset, train the churn model and launch the analytical applications.

## Environment

- Python 3.10 or newer
- Install dependencies with `python -m pip install -r requirements.txt`

## Generate the customer dataset

From the repository root:

```bash
python scripts/generate_dataset.py
```

The script writes `data/raw/churn_customers_synthetic.csv` using a fixed default seed of 42. The output location is resolved from the script's own repository path, so the command also works when invoked from another working directory.

Optional arguments:

```bash
python scripts/generate_dataset.py --seed 42 --rows 20000
python scripts/generate_dataset.py --output /path/to/customer_data.csv
```

The generator creates an underlying population and injects repeatable data-quality conditions, including missing values, duplicates, inconsistent segment labels, implausible ages and outliers.

## Train and evaluate

```bash
python scripts/train_pipeline.py
```

The pipeline validates and cleans the source file, creates a stratified train/test split, compares baseline and candidate models, fits the selected Logistic Regression pipeline, evaluates the held-out sample and exports model metadata, predictions and cleaned data.

Key outputs:

- `models/logistic_regression_pipeline.joblib`
- `data/processed/churn_predictions.csv`
- `data/processed/churn_customers_clean.csv`

The random state defaults to 42. Use `--help` to review available training options.

## Launch the applications

Interactive analytics and retention workspace:

```bash
streamlit run dashboard/streamlit_command_centre.py
```

Single and batch prediction interface:

```bash
streamlit run app.py
```

## Run automated tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

The test suite covers data validation, feature engineering, model behaviour and reproducible dataset generation.

## Power BI

See [dashboard/powerbi_dashboard_guide.md](dashboard/powerbi_dashboard_guide.md) for the six-page Power BI design, data model, DAX measures and implementation notes. The Streamlit command centre is the runnable interactive dashboard included in this repository.

## Interpretation and governance

- Review recall, precision, balanced accuracy, ROC-AUC and calibration together; accuracy alone can conceal poor minority-class detection.
- Choose outreach thresholds using the available team capacity and the relative cost of missed churners and unnecessary contacts.
- Treat feature importance as predictive association, not causal evidence.
- Review retention recommendations before customer contact.
- Validate model behaviour, fairness, calibration and drift before any real operational use.
- The supplied data and bank narrative are illustrative portfolio materials, not live customer records or operational bank data.
