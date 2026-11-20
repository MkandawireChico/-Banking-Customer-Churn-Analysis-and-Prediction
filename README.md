# Banking Customer Intelligence & Retention Command Centre

An end-to-end banking customer analytics project that connects data quality, churn-risk modelling, model governance and practical retention operations in a Malawi-focused commercial banking scenario.

<p align="center">
  <a href="https://banking-customer-churn-command-centre.streamlit.app/"><strong>Open the Live Command Centre →</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Command%20Centre-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--learn-Modelling-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/SQL-SQLite-003B57?logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/Currency-MWK-2B8C82" alt="Malawi Kwacha">
</p>

## The business problem

Customer attrition can reduce recurring revenue and weaken long-term relationships. A useful retention workflow needs more than a churn score: teams need portfolio visibility, interpretable risk signals, defensible model evaluation, prioritised outreach and a way to organise follow-up work.

This project demonstrates that workflow from data preparation through operational decision support.

## Explore the project

| Resource | Purpose |
|---|---|
| [Live Streamlit Command Centre](https://banking-customer-churn-command-centre.streamlit.app/) | Interactive portfolio intelligence and retention operations |
| [Analysis notebook](notebooks/customer_churn_analysis.ipynb) | Reproducible exploratory analysis and review of saved model evidence |
| [Project walkthrough](docs/PROJECT_WALKTHROUGH.md) | Business workflow, architecture and interpretation |
| [Reproducibility guide](REPRODUCIBILITY.md) | Environment, data generation, model refresh and artefacts |
| [Power BI guide](dashboard/powerbi_dashboard_guide.md) | Optional specification for a separate BI implementation |
| [Source modules](src/) | Validation, feature engineering, modelling, explainability, retention and case management |
| [Automated tests](tests/) | Unit and pipeline checks |

## The Command Centre

The deployed Streamlit application is a light-theme operational workspace. It includes:

- **Executive Overview** — portfolio KPIs, churn exposure, risk distribution and branch/segment views.
- **Customer Intelligence** — interactive customer, branch, segment and risk exploration.
- **Retention Prioritisation** — configurable capacity, ranked outreach queues and downloadable worklists.
- **Prediction Lab** — batch customer scoring from an uploaded CSV.
- **Model Governance** — held-out performance, threshold trade-offs, calibration and subgroup diagnostics.
- **Retention Campaign Studio** — campaign setup, manager allocation and execution-list export.
- **Retention Case Management** — editable case register, follow-up monitoring and CSV handoff.

Financial values in the application are presented in Malawi Kwacha (MWK).

**Operational boundary:** campaign assignments and case edits are CSV-based handoffs. They are not written to a shared CRM or permanent multi-user database.

## Modelling and governance

The training workflow compares a majority-class baseline, Logistic Regression and Random Forest using a validation split drawn only from the training partition. Logistic Regression is retained as the initial operating candidate for recall-oriented retention screening; this is a documented starting choice, not a claim that it is universally best.

The selected model is wrapped in sigmoid probability calibration. The calibrator is fitted using five-fold cross-validation within the training partition. The independent final test partition is excluded from candidate selection, threshold design and calibrator fitting, and is used for the final held-out evaluation.

Evaluation considers accuracy, precision, recall, F1, ROC-AUC, average precision, Brier score and log loss. Calibration diagnostics include a reliability curve and expected calibration error (ECE). If calibration is weak, the application warns that scores are more appropriate for relative ranking than as precise individual probabilities.

The latest metrics are stored with the trained model artefact and are displayed dynamically in the application and notebook. They are intentionally not hard-coded here because they change when the controlled full-population refresh is run.

### Keep evaluation populations separate

- `data/processed/churn_predictions.csv` contains the independent held-out test cohort and actual outcomes. Use it for model evaluation.
- `data/processed/churn_customer_scores.csv` contains scores for the full cleaned customer population. Use it for portfolio monitoring and prioritisation, not for independent performance claims.
- `data/processed/churn_customers_clean.csv` contains the cleaned customer profile data.

## Selected analytical themes

The exploratory work examines transaction recency, complaints, product depth, digital engagement, customer segment and branch. These patterns are descriptive associations in a generated dataset; they do not establish that changing a feature will cause a customer to remain.

Charts are available in [visuals/charts](visuals/charts/), including customer behaviour, confusion matrices, ROC/precision-recall curves, threshold analysis and feature importance.

## Repository structure

```text
├── .github/workflows/       # CI and controlled model refresh
├── dashboard/
│   ├── streamlit_command_centre.py
│   └── powerbi_dashboard_guide.md
├── data/
│   ├── raw/                 # Generated source cohort
│   └── processed/           # Clean data, test predictions, portfolio scores
├── docs/                    # Project walkthrough
├── models/                  # Current calibrated production pipeline
├── notebooks/               # Exploratory analysis and model review
├── scripts/                 # Dataset generation and training pipeline
├── sql/                     # SQL analysis examples and SQLite database
├── src/                     # Reusable project modules
├── tests/                   # Automated tests
├── visuals/charts/          # Exported analytical charts
├── requirements.txt
└── REPRODUCIBILITY.md
```

## Run locally

```bash
git clone https://github.com/MkandawireChico/-Banking-Customer-Churn-Analysis-and-Prediction.git
cd -Banking-Customer-Churn-Analysis-and-Prediction
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt

python scripts/generate_dataset.py
python scripts/train_pipeline.py
pytest tests/ -v
streamlit run dashboard/streamlit_command_centre.py
```

The notebook is a review and exploration layer over the canonical generated artefacts. Run the generation and training scripts first, then open `notebooks/customer_churn_analysis.ipynb`.

### Controlled production refresh

The repository includes a manual GitHub Actions workflow for refreshing the model and its generated artefacts. It requires an explicit confirmation and is not an automatic retrain on every code push.

[Open the model refresh workflow](https://github.com/MkandawireChico/-Banking-Customer-Churn-Analysis-and-Prediction/actions/workflows/refresh-production-model.yml)

## Responsible use and limitations

The customer records and churn outcomes are generated for a controlled analytics demonstration. They are not real customer accounts, operational bank records or evidence of actual Nyanja Commercial Bank performance.

Before any real banking use, the approach would require authorised representative data, privacy and access controls, data-lineage checks, fairness assessment, drift and calibration monitoring, intervention testing and accountable human review. A churn score must not be used by itself to deny services, close accounts or apply punitive treatment.

## Author

**Chico Mkandawire**  
MSc Data Science, Malawi University of Science and Technology (in progress)  
BSc Management Information Systems, Malawi University of Business and Applied Sciences

---

<p align="center"><strong>From customer intelligence to responsible retention action.</strong></p>
