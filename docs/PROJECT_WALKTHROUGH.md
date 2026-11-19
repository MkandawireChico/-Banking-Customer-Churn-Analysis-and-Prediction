# Project Walkthrough

## Executive summary

This project demonstrates how a banking analytics team can move from customer data to a structured churn-retention workflow. It combines exploratory analysis, predictive modelling, model interpretation and operational prioritisation in a reproducible Python project.

A probability score is only one part of a useful retention process. Decision-makers also need segment context, an understanding of model trade-offs, a ranked list that respects team capacity and governance controls around customer treatment.

## Business workflow

```text
Customer cohort
      |
      v
Data validation and cleaning
      |
      v
Feature engineering
      |
      v
Train / validation / test workflow
      |
      v
Model comparison and evaluation
      |
      +----------------------------+
      |                            |
      v                            v
Full-cohort scoring          Held-out test evaluation
      |                            |
      v                            v
Risk bands and reason codes   Model performance evidence
      |
      v
Capacity-aware retention queue
      |
      v
Human review and intervention
```

The full-cohort score output and held-out evaluation output are intentionally separate. This avoids presenting training-cohort scores as independent model validation.

## Application workspaces

### Executive Overview
Portfolio KPIs, observed churn, risk distribution, high-risk balance exposure, branch and segment comparisons, inactivity analysis and a priority watchlist.

### Customer Intelligence
Interactive segment, branch, risk and minimum-score filters, behavioural analysis, customer-level exploration and filtered export.

### Retention Prioritisation
A configurable worklist with adjustable outreach capacity, eligible risk bands and minimum score. The queue is intended for relationship-team review, not automatic customer contact.

### Prediction Lab
Upload a new customer CSV, validate required model features, score records and export a ranked result.

### Model Governance
Held-out metrics, threshold trade-offs and operating safeguards covering fairness, data quality, monitoring and intervention testing.

## Model selection and interpretation

The candidate models are a majority-class baseline, Logistic Regression and Random Forest. Logistic Regression is selected for its stronger recall and ROC-AUC in the recorded held-out evaluation.

| Metric | Logistic Regression |
|---|---:|
| ROC-AUC | 0.7417 |
| Recall | 0.6529 |
| Precision | 0.2271 |
| F1 | 0.3370 |
| Balanced accuracy | 0.6670 |

Recall measures the share of actual churners identified by the model. Precision measures the share of flagged customers who actually churned. Lower precision means the model will flag customers who do not churn; this is an important operational cost. Threshold selection should reflect outreach capacity, intervention cost and the relative cost of missed churners.

## Analytical signals

The exploratory analysis identifies transaction recency, complaints and product depth as useful dimensions for further investigation. High balance alone does not appear to distinguish churn risk in the analysed cohort.

These are descriptive associations, not proof that changing any one factor will prevent churn. Retention actions should be tested with controlled experiments and assessed for customer outcomes as well as commercial value.

## Data and reproducibility

The project uses a reproducible generated customer cohort. The generation and training scripts are:

- `scripts/generate_dataset.py`
- `scripts/train_pipeline.py`

Generated artefacts include a cleaned customer dataset, held-out test predictions, full-cohort scores and a trained model with metadata. Run instructions and deterministic settings are documented in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

```bash
pip install -r requirements.txt
python scripts/generate_dataset.py
python scripts/train_pipeline.py
pytest tests/ -v
streamlit run dashboard/streamlit_command_centre.py
```

## Production-readiness considerations

A real banking deployment would require more than a successful model run:

- Confirm lawful basis, consent and access controls for customer data.
- Validate data lineage, freshness, schema and quality.
- Assess discrimination, calibration and error rates across relevant customer groups.
- Define model ownership, approval, monitoring and rollback procedures.
- Test interventions and measure incremental retention impact.
- Keep a human decision-maker accountable for customer-facing actions.

The model is a prioritisation aid. It should not be used by itself to close accounts, deny products or apply punitive treatment.

## Suggested portfolio presentation

1. Open the [live command centre](https://banking-customer-churn-command-centre.streamlit.app/) and start with Executive Overview.
2. Demonstrate segment and branch filtering in Customer Intelligence.
3. Change outreach capacity in Retention Prioritisation and show how the queue responds.
4. Use Prediction Lab to demonstrate batch scoring with a correctly structured CSV.
5. Finish in Model Governance with held-out metrics and threshold trade-offs.
6. Return to the repository to inspect the modular pipeline, tests and reproducibility instructions.

This walkthrough connects the business question, analytical evidence, operational workflow and engineering implementation.
