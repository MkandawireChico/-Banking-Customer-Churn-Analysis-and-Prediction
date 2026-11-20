# Project Walkthrough

## Executive summary

The Banking Customer Intelligence & Retention Command Centre demonstrates an end-to-end decision-support workflow for a Malawi-focused commercial banking scenario. It brings together customer data quality, exploratory analysis, leakage-aware churn modelling, model governance and practical retention operations.

The central idea is that a churn score is only one input. A useful workflow also needs a clear view of portfolio exposure, a ranked worklist, relationship-manager capacity, case follow-up and safeguards around interpretation.

## End-to-end workflow

```text
Generated customer cohort
          |
          v
Data validation and cleaning
          |
          v
Leakage-aware feature pipeline
          |
          v
Train / validation / independent test
          |
          v
Candidate comparison on validation data
          |
          v
Calibrated production pipeline
          |
          +------------------------------+
          |                              |
          v                              v
Full-population scoring          Held-out test evaluation
          |                              |
          v                              v
Portfolio risk and queue         Model governance evidence
          |
          v
Campaign planning and case follow-up
          |
          v
Human review and intervention
```

The two scoring outputs have different purposes. The held-out test predictions support independent evaluation. Full-population scores support portfolio exploration and retention prioritisation; they must not be used to claim independent test performance.

## Command Centre workspaces

- **Executive Overview:** portfolio KPIs, churn exposure, risk distribution and branch/segment comparisons.
- **Customer Intelligence:** customer-level exploration with interactive filters.
- **Retention Prioritisation:** configurable capacity and ranked outreach queue.
- **Prediction Lab:** batch scoring for an uploaded customer file.
- **Model Governance:** held-out metrics, threshold trade-offs, calibration and subgroup diagnostics.
- **Retention Campaign Studio:** campaign filters, contact planning, relationship-manager allocation and execution CSV export.
- **Retention Case Management:** starter case register, editable case fields, overdue follow-up monitoring and CSV import/export.

The campaign and case workspaces are operational planning tools. CSV import/export is the persistence handoff; the application does not currently write to a shared CRM or database.

## Model development and evaluation

The modular training script compares a majority-class baseline, Logistic Regression and Random Forest using a validation split taken from the training partition. Logistic Regression is retained as the initial operating candidate because the current use case values recall-oriented screening. That decision is provisional and should be revisited using intervention cost, outreach capacity and the cost of false positives and false negatives.

A sigmoid calibrator is fitted with five-fold cross-validation using the full training partition. The independent test partition remains untouched until final evaluation. The saved metadata records classification, discrimination and calibration metrics, validation-only threshold analysis, capacity scenarios and subgroup diagnostics.

The current metrics are loaded dynamically from the model artefact. They are deliberately not copied into this document because the controlled refresh workflow can regenerate the model and its evaluation results.

## Exploratory signals

The notebook and exported charts examine transaction recency, complaints, product depth, digital engagement, segment and branch. These relationships are descriptive associations in the generated cohort. They are not evidence that changing a particular behaviour will cause retention.

## Repository architecture

- `scripts/generate_dataset.py` creates the reproducible synthetic source cohort.
- `scripts/train_pipeline.py` is the canonical model-training and artefact-generation workflow.
- `src/` contains reusable validation, feature engineering, modelling, explainability, retention and case-management logic.
- `dashboard/streamlit_command_centre.py` provides the interactive application.
- `notebooks/customer_churn_analysis.ipynb` provides exploratory analysis and a review of the saved model evidence; it does not duplicate model training.
- `tests/` provides automated quality checks.
- `.github/workflows/` contains continuous integration and the manually triggered production refresh workflow.

## Run locally

```bash
pip install -r requirements.txt
python scripts/generate_dataset.py
python scripts/train_pipeline.py
pytest tests/ -v
streamlit run dashboard/streamlit_command_centre.py
```

See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for the full procedure and evaluation boundaries.

## Production-readiness considerations

A real banking deployment would additionally require authorised representative customer data, privacy and access controls, validated data lineage, fairness assessment, model approval, drift and calibration monitoring, rollback procedures and controlled experiments to measure retention impact. Human decision-makers must remain accountable for customer-facing actions.

A churn score must not be used by itself to deny services, close accounts or apply punitive treatment.

## Suggested portfolio demonstration

1. Start with the live Command Centre's Executive Overview.
2. Filter Customer Intelligence by branch, segment and risk.
3. Change outreach capacity in Retention Prioritisation.
4. Build a campaign and demonstrate relationship-manager allocation.
5. Open Retention Case Management to show follow-up tracking and CSV handoff.
6. Finish with Model Governance and explain why test evaluation, calibration and threshold trade-offs matter.
7. Return to the repository to inspect the notebook, modular pipeline and automated tests.
