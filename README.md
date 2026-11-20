# Banking Customer Intelligence & Retention Command Centre

**An end-to-end customer analytics, churn-risk and retention operations project for a Malawi-focused commercial banking scenario.**

<p align="center">
  <a href="https://banking-customer-churn-command-centre.streamlit.app/"><strong>OPEN THE LIVE RETENTION COMMAND CENTRE →</strong></a>
</p>
<p align="center">
  <a href="https://colab.research.google.com/github/MkandawireChico/-Banking-Customer-Churn-Analysis-and-Prediction/blob/main/notebooks/customer_churn_analysis.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--learn-Modelling-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/SQLite-SQL-003B57?logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/Currency-MWK-2B8C82" alt="Malawi Kwacha">
</p>

---

## 1. Project at a glance

Customer churn is not only a classification problem. A bank needs to understand where attrition exposure is concentrated, identify customers who may need attention, decide how to use limited relationship-manager capacity and track what happens after outreach.

This project brings those stages together in a working analytics and decision-support environment. It combines a reproducible data pipeline, leakage-aware model development, independent model evaluation, full-population risk scoring and operational retention tools.

The project is presented through **Nyanja Commercial Bank**, a fictional Malawi-focused banking scenario. Monetary values in the application and analysis are expressed in **Malawi Kwacha (MWK)**.

| Project area | What is delivered |
|---|---|
| Data quality | Validation, cleaning, schema checks and feature preparation |
| Customer analytics | Portfolio, segment, branch and behavioural exploration |
| Predictive modelling | Baseline comparison, Logistic Regression and Random Forest evaluation |
| Model governance | Held-out evaluation, threshold analysis, calibration and subgroup diagnostics |
| Risk prioritisation | Risk bands, ranked customer queues and capacity-aware selection |
| Campaign operations | Campaign filters, manager allocation and execution-list export |
| Case management | Case status, ownership, follow-up monitoring and CSV handoff |
| Deployment | Interactive Streamlit Command Centre |

## 2. Explore the project

- **[Launch the live Streamlit Command Centre](https://banking-customer-churn-command-centre.streamlit.app/)** — explore the interactive application.
- **[Open the analysis notebook in Google Colab](https://colab.research.google.com/github/MkandawireChico/-Banking-Customer-Churn-Analysis-and-Prediction/blob/main/notebooks/customer_churn_analysis.ipynb)** — execute the analysis and render its charts.
- **[View the notebook on GitHub](notebooks/customer_churn_analysis.ipynb)** — inspect the source notebook.
- **[Project walkthrough](docs/PROJECT_WALKTHROUGH.md)** — understand the architecture and business workflow.
- **[Reproducibility guide](REPRODUCIBILITY.md)** — setup, artefact generation and refresh procedures.
- **[Power BI implementation guide](dashboard/powerbi_dashboard_guide.md)** — optional specification for a separate BI layer.
- **[Source modules](src/)** · **[Automated tests](tests/)** · **[SQL examples](sql/)**

## 3. The Retention Command Centre

The deployed Streamlit application is designed as a light-theme decision workspace, not just a model demonstration.

### Executive Overview
Presents portfolio-level indicators, churn exposure, risk distribution and breakdowns by customer segment and branch. It gives decision-makers an initial view of where attention may be required.

### Customer Intelligence
Supports exploration of customers using segment, branch and risk filters. It connects customer-level records with behavioural and portfolio context.

### Retention Prioritisation
Creates a ranked worklist based on churn risk and configurable outreach capacity. The aim is to help teams focus limited contact resources rather than treating every customer as equally urgent.

### Prediction Lab
Accepts a customer CSV for batch scoring through the saved production pipeline and provides risk assessments for the submitted records.

### Model Governance
Surfaces held-out model performance, threshold trade-offs, probability calibration and subgroup diagnostics. It is intended to make model limitations visible alongside performance.

### Retention Campaign Studio
Allows a user to define a campaign, select eligible customers, set a contact limit and distribute the ranked queue across relationship managers. It provides workload summaries and an execution-list CSV export.

### Retention Case Management
Provides a case register for ownership, status, contact channel, latest outcome, notes and next follow-up date. It highlights overdue follow-ups and connects case records to current customer risk context.

**Important operational boundary:** Campaign Studio and Case Management currently use CSV import/export as a persistence handoff. The application does not write changes to a shared CRM or permanent multi-user database. Case and campaign records therefore need controlled storage and access outside the app.

## 4. Analytical workflow

The repository separates reusable production code from the explanatory notebook.

1. **Generate the cohort** — create a reproducible synthetic customer dataset for the banking scenario.
2. **Validate and clean** — inspect schema, missingness, duplicates, category consistency and implausible values.
3. **Engineer features** — prepare behavioural and relationship features for modelling.
4. **Compare candidate models** — evaluate a majority-class baseline, Logistic Regression and Random Forest using training/validation data.
5. **Select and calibrate** — document the operating candidate and fit sigmoid calibration within the training partition.
6. **Evaluate independently** — reserve the final test partition for held-out performance and calibration assessment.
7. **Explain and segment risk** — review feature importance, risk bands and threshold/capacity trade-offs.
8. **Score the portfolio** — generate full-population scores for monitoring and prioritisation.
9. **Operationalise the results** — use the Streamlit workspaces to explore, prioritise, plan campaigns and manage follow-up cases.

The training pipeline in `scripts/train_pipeline.py` is the canonical source for model fitting and evaluation. The notebook reads the generated artefacts; it does not create a competing training implementation.

## 5. Modelling approach and governance

The project compares a majority-class baseline with Logistic Regression and Random Forest. Candidate selection and threshold design use a validation split drawn from the training partition. The independent final test set is not used to select the model, tune its threshold or fit its calibrator.

Logistic Regression is retained as the initial recall-oriented operating candidate for retention screening. This is a documented starting point, not a claim that it is universally superior. A bank's final operating choice should consider intervention costs, the cost of missed churners, contact capacity and the quality of the resulting customer experience.

The production pipeline applies **sigmoid probability calibration** using five-fold cross-validation within the training partition. The final held-out test partition is kept separate for independent evaluation.

Evaluation considers multiple dimensions:

- **Discrimination:** ROC-AUC and average precision.
- **Classification trade-offs:** precision, recall, F1 and accuracy.
- **Probability quality:** Brier score, log loss, reliability curve and expected calibration error (ECE).
- **Decision support:** threshold scenarios and capacity-based workload implications.
- **Responsible interpretation:** subgroup metrics and minimum-sample safeguards.

No single metric should be treated as a complete measure of business value. In particular, a high recall can identify more potential churners while also creating a larger outreach workload and more false positives.

### Two populations, two purposes

The repository deliberately distinguishes the held-out evaluation cohort from the scored portfolio:

| Artefact | Purpose | Appropriate use |
|---|---|---|
| `data/processed/churn_predictions.csv` | Independent held-out test cohort with actual outcomes | Model performance and risk-band validation |
| `data/processed/churn_customer_scores.csv` | Scores across the cleaned customer population | Portfolio monitoring and retention prioritisation |
| `data/processed/churn_customers_clean.csv` | Cleaned customer profile data | Descriptive analysis and customer context |

Do not calculate independent model-performance claims from the full-population score file: it can include customers used to fit the model.

Latest metrics are loaded from the current model artefacts rather than copied into this README. They can change after the controlled model-refresh workflow is run.

## 6. Exploratory analysis

The notebook investigates customer and portfolio patterns across:

- Customer segment and branch.
- Transaction recency and inactivity.
- Complaint counts.
- Number of products held.
- Customer-level churn outcomes.
- Model discrimination and classification errors.
- Risk-band separation on the held-out cohort.
- Probability calibration and threshold/capacity scenarios.

The analysis is descriptive. Associations between inactivity, complaints, product depth and churn do not establish that changing any one factor will prevent attrition. Intervention effectiveness would need to be tested separately.

Charts are generated by executing the notebook. The repository also contains exported figures under [visuals/charts](visuals/charts/).

## 7. Run the notebook in Google Colab

The notebook is designed to be executed. Its charts and tables are not visible in a fresh GitHub preview until the cells have been run and outputs saved.

1. Select the **Open in Colab** badge at the top of this README.
2. In Colab, choose **Runtime → Run all**.
3. The notebook setup cell clones the repository and installs the project requirements.
4. The analysis cells load the current committed data and model artefacts, then render tables and charts.
5. If a required artefact is missing, follow the repository's controlled refresh instructions rather than silently substituting a different evaluation population.
6. After successful execution, use **File → Save a copy in GitHub** (or download the `.ipynb` and commit it through a pull request) to preserve the executed notebook.

Use a Colab runtime with sufficient memory. A fresh execution can take time because dependencies must be installed and the notebook reads the project datasets and model artefacts.

## 8. Run the project locally

Clone the repository and install its dependencies:

```bash
git clone https://github.com/MkandawireChico/-Banking-Customer-Churn-Analysis-and-Prediction.git
cd -Banking-Customer-Churn-Analysis-and-Prediction
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
```

Generate the dataset and model artefacts when setting up a fresh local environment:

```bash
python scripts/generate_dataset.py
python scripts/train_pipeline.py
pytest tests/ -v
```

Launch the application:

```bash
streamlit run dashboard/streamlit_command_centre.py
```

Open `notebooks/customer_churn_analysis.ipynb` in Jupyter after the required artefacts have been generated.

### Controlled production refresh

The repository also contains a **manual GitHub Actions workflow** for refreshing the production model and generated artefacts. It requires an explicit confirmation and is not an automatic retrain on every push.

[Open the production model refresh workflow](https://github.com/MkandawireChico/-Banking-Customer-Churn-Analysis-and-Prediction/actions/workflows/refresh-production-model.yml)

## 9. Repository map

```text
.github/workflows/   CI checks and controlled model refresh
dashboard/           Streamlit application and Power BI guide
data/raw/            Generated source cohort
data/processed/      Clean profiles, held-out predictions and portfolio scores
docs/                Project walkthrough and supporting documentation
models/              Saved production model and evaluation metadata
notebooks/           Exploratory analysis and model-governance review
scripts/             Dataset generation and canonical training pipeline
sql/                 SQL analysis examples and SQLite database
src/                 Validation, features, modelling, explainability and retention logic
tests/               Automated tests and pipeline checks
visuals/charts/      Exported analytical charts
```

## 10. Responsible use and limitations

- The customer records and churn outcomes are synthetic. They are not real customer accounts, operational bank records or evidence of actual Nyanja Commercial Bank performance.
- Model scores are decision-support signals, not certainty about an individual's future behaviour.
- Group-level associations are not causal findings.
- Thresholds should reflect intervention cost, contact capacity and the consequences of false positives and false negatives.
- Campaign and case workflows are CSV-based and do not yet provide shared CRM persistence, audit-grade contact history or multi-user concurrency.
- Before real banking use, the approach would require authorised representative data, privacy and access controls, data-lineage checks, fairness assessment, drift and calibration monitoring, intervention testing and accountable human review.
- A churn score must not independently trigger denial of service, account closure or punitive treatment.

## 11. Author

**Chico Mkandawire**  
MSc Data Science, Malawi University of Science and Technology (in progress)  
BSc Management Information Systems, Malawi University of Business and Applied Sciences

---

<p align="center"><strong>From customer intelligence to responsible retention action.</strong></p>
