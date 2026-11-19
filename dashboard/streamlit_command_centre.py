"""Interactive banking customer retention command centre."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from src.modeling import ChurnModelPipeline
from src.retention import RetentionRecommender

st.set_page_config(page_title="Banking Customer Retention Command Centre", page_icon="🏦", layout="wide")
NAVY = "#142B4A"


@st.cache_data
def load_data():
    score_file = ROOT / "data" / "processed" / "churn_customer_scores.csv"
    test_file = ROOT / "data" / "processed" / "churn_predictions.csv"
    customer_file = ROOT / "data" / "processed" / "churn_customers_clean.csv"
    if not customer_file.exists():
        return None
    if score_file.exists():
        scores = pd.read_csv(score_file)
        score_scope = "Full population"
    else:
        # Streamlit deployments may not have the optional full-population
        # score export. Recreate it from the committed model and cleaned
        # customer cohort so the retention queue still covers every customer.
        model, _ = load_model()
        if model is not None:
            features = customers[model.feature_names_in_]
            probabilities = model.predict_proba(features)[:, 1]
            scores = pd.DataFrame({
                "Customer_ID": customers["Customer_ID"].values,
                "Churn_Probability": probabilities,
                "Predicted_Churn_0.50": (probabilities >= 0.50).astype(int),
            })
            scores["Risk_Band"] = scores["Churn_Probability"].apply(
                RetentionRecommender().assign_risk_band
            )
            score_scope = "Full population (scored at runtime)"
        elif test_file.exists():
            scores = pd.read_csv(test_file)
            score_scope = "Held-out test cohort (fallback)"
        else:
            return None
    customers = pd.read_csv(customer_file).drop_duplicates(subset=["Customer_ID"], keep="first")
    data = scores.merge(customers, on="Customer_ID", how="left", suffixes=("", "_profile"))
    data["Score_Scope"] = score_scope
    if "Actual_Churn" not in data and "Churn" in data:
        data["Actual_Churn"] = data["Churn"]
    if "Risk_Band" not in data:
        data["Risk_Band"] = data["Churn_Probability"].apply(
            RetentionRecommender().assign_risk_band
        )
    return data


@st.cache_resource
def load_model():
    path = ROOT / "models" / "logistic_regression_pipeline.joblib"
    if not path.exists():
        return None, {}
    payload = joblib.load(path)
    return ChurnModelPipeline.load(path), payload.get("metadata", {})


def money(value):
    return f"MWK {value:,.0f}"


def overview(data, metadata):
    st.title("Banking Customer Retention Command Centre")
    st.caption("Portfolio intelligence • Customer risk • Retention prioritisation")
    if data is None or data.empty:
        st.warning("Analytics exports are missing. Run scripts/train_pipeline.py from the repository root.")
        return
    score_scope = data["Score_Scope"].iloc[0] if "Score_Scope" in data else "Unknown"
    if score_scope == "Held-out test cohort (fallback)":
        st.info("This checkout does not yet contain full-population customer scores. The dashboard is using the held-out test cohort; rerun scripts/train_pipeline.py to populate the complete retention queue.")
    else:
        st.caption("Portfolio summaries use full-population scores; held-out model metrics are reported separately. Portfolio scores support prioritisation and are not independent model evaluation.")
    total = len(data)
    churn = data["Actual_Churn"].mean()
    high_mask = data["Risk_Band"] == "High Risk"
    high_count = int(high_mask.sum())
    balance_exposure = data.loc[high_mask, "Average_Monthly_Balance"].sum() if "Average_Monthly_Balance" in data else 0
    c = st.columns(4)
    c[0].metric("Customers scored", f"{total:,}")
    c[1].metric("Historical churn rate", f"{churn:.1%}")
    c[2].metric("High-risk customers", f"{high_count:,}", f"{high_count / total:.1%} of evaluated")
    c[3].metric("High-risk customer balances", money(balance_exposure))
    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Observed churn by customer segment")
        if "Customer_Segment" in data:
            st.bar_chart(data.groupby("Customer_Segment", observed=True)["Actual_Churn"].mean().sort_values())
        st.subheader("Observed churn by branch")
        if "Branch_Location" in data:
            st.bar_chart(data.groupby("Branch_Location", observed=True)["Actual_Churn"].mean().sort_values())
    with right:
        st.subheader("Portfolio risk distribution")
        risk = data["Risk_Band"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(risk.values, labels=risk.index, autopct="%1.1f%%", startangle=90,
               colors=["#69A88D", "#E4B65C", "#C96A67"], wedgeprops={"width": 0.42})
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.subheader("Churn by inactivity band")
        if "Days_Since_Last_Transaction" in data:
            bins = pd.cut(data["Days_Since_Last_Transaction"], [-1, 30, 60, 90, 180, float("inf")],
                          labels=["0–30", "31–60", "61–90", "91–180", "180+"])
            st.bar_chart(data.assign(Inactivity_Band=bins).groupby("Inactivity_Band", observed=True)["Actual_Churn"].mean())
    metrics = metadata.get("test_metrics", {})
    if metrics:
        st.divider()
        st.subheader("Held-out model validation")
        c = st.columns(5)
        for col, key, label in zip(c, ["roc_auc", "recall", "precision", "f1", "balanced_accuracy"],
                                   ["ROC-AUC", "Recall", "Precision", "F1", "Balanced accuracy"]):
            value = metrics.get(key)
            col.metric(label, f"{value:.1%}" if isinstance(value, (int, float)) else "—")


def customer_intelligence(data):
    st.title("Customer Intelligence")
    if data is None:
        st.warning("Run the training pipeline to generate the customer analysis exports.")
        return
    cols = st.columns(3)
    segments = sorted(data["Customer_Segment"].dropna().unique()) if "Customer_Segment" in data else []
    branches = sorted(data["Branch_Location"].dropna().unique()) if "Branch_Location" in data else []
    chosen_segments = cols[0].multiselect("Customer segment", segments, default=segments)
    chosen_branches = cols[1].multiselect("Branch", branches, default=branches)
    chosen_risk = cols[2].multiselect("Risk band", ["Low Risk", "Medium Risk", "High Risk"],
                                      default=["Low Risk", "Medium Risk", "High Risk"])
    view = data.copy()
    if "Customer_Segment" in view:
        view = view[view["Customer_Segment"].isin(chosen_segments)]
    if "Branch_Location" in view:
        view = view[view["Branch_Location"].isin(chosen_branches)]
    view = view[view["Risk_Band"].isin(chosen_risk)]
    st.caption(f"{len(view):,} customers match the selected filters.")
    left, right = st.columns(2)
    with left:
        st.subheader("Churn by complaint count")
        if "Number_of_Complaints" in view:
            temp = view.assign(Complaint_Group=view["Number_of_Complaints"].clip(upper=4))
            st.bar_chart(temp.groupby("Complaint_Group", observed=True)["Actual_Churn"].mean())
    with right:
        st.subheader("Churn by product holding")
        if "Number_of_Products" in view:
            st.bar_chart(view.groupby("Number_of_Products", observed=True)["Actual_Churn"].mean())
    display = [x for x in ["Customer_ID", "Customer_Segment", "Branch_Location", "Churn_Probability",
                           "Risk_Band", "Days_Since_Last_Transaction", "Number_of_Complaints",
                           "Number_of_Products", "Average_Monthly_Balance"] if x in view]
    st.dataframe(view[display].sort_values("Churn_Probability", ascending=False), use_container_width=True, hide_index=True)


def retention_prioritisation(data):
    st.title("Retention Prioritisation")
    st.caption("Rank customers by churn probability and review a suggested next-best action.")
    if data is not None and "Score_Scope" in data and data["Score_Scope"].iloc[0] != "Full population":
        st.info("The queue currently covers only the held-out test cohort. Run the latest training pipeline to generate full-population customer scores.")
    if data is None:
        st.warning("Prediction exports are missing. Run the training pipeline first.")
        return
    capacity = st.slider("Relationship-team outreach capacity", 25, 2000, 500, 25)
    queue = RetentionRecommender().create_outreach_queue(data, capacity=capacity)
    if queue.empty:
        st.info("No medium- or high-risk customers are available in this sample.")
        return
    c = st.columns(3)
    c[0].metric("Customers in queue", f"{len(queue):,}")
    c[1].metric("Critical priority", int((queue["priority"] == "Critical").sum()))
    c[2].metric("High priority", int((queue["priority"] == "High").sum()))
    fields = [x for x in ["Outreach_Rank", "Customer_ID", "Risk_Band", "Churn_Probability", "priority",
                          "recommended_action", "Days_Since_Last_Transaction", "Number_of_Complaints",
                          "Customer_Segment", "Branch_Location", "Average_Monthly_Balance"] if x in queue]
    st.dataframe(queue[fields], use_container_width=True, hide_index=True)
    st.download_button("Download outreach queue", queue.to_csv(index=False).encode("utf-8"),
                       "banking_retention_outreach_queue.csv", "text/csv")


def prediction_lab():
    st.title("Prediction Lab")
    model, _ = load_model()
    if model is None:
        st.warning("Trained model not found. Run scripts/train_pipeline.py first.")
        return
    upload = st.file_uploader("Upload a customer CSV for batch scoring", type=["csv"])
    st.caption("The upload must contain the raw input fields used during model training.")
    if upload is None:
        return
    try:
        customers = pd.read_csv(upload)
        missing = sorted(set(model.feature_names_in_) - set(customers.columns))
        if missing:
            st.error("Missing required fields: " + ", ".join(missing))
            return
        scored = customers.copy()
        scored["Churn_Probability"] = model.predict_proba(customers)[:, 1]
        scored["Risk_Band"] = scored["Churn_Probability"].map(RetentionRecommender().assign_risk_band)
        scored["Predicted_Churn_0.50"] = (scored["Churn_Probability"] >= 0.50).astype(int)
        st.success(f"Scored {len(scored):,} customer records.")
        st.dataframe(scored.sort_values("Churn_Probability", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("Download scored customers", scored.to_csv(index=False).encode("utf-8"),
                           "banking_customer_churn_scores.csv", "text/csv")
    except Exception as exc:
        st.error(f"Unable to score this file: {exc}")


def model_governance(metadata):
    st.title("Model Performance & Governance")
    if not metadata:
        st.warning("Model metadata is unavailable. Train the model to generate an evaluation record.")
        return
    metrics = metadata.get("test_metrics", {})
    labels = {"roc_auc": "ROC-AUC", "average_precision": "Average precision", "accuracy": "Accuracy",
              "balanced_accuracy": "Balanced accuracy", "precision": "Precision", "recall": "Recall",
              "f1": "F1 score", "specificity": "Specificity", "brier_score": "Brier score", "log_loss": "Log loss"}
    st.subheader("Held-out test performance")
    st.dataframe(pd.DataFrame([{"Metric": label, "Value": metrics.get(key)}
                               for key, label in labels.items() if key in metrics]), hide_index=True)
    ece = metadata.get("calibration_ece")
    if ece is not None:
        st.metric("Expected calibration error", f"{ece:.3f}")
    st.subheader("Operating principles")
    st.markdown("- Scores support prioritisation; they do not prove a customer will leave.\n"
                "- Review reason codes and retention actions before any customer contact.\n"
                "- Monitor calibration, data quality, drift and performance after each model refresh.\n"
                "- Set thresholds using contact capacity, intervention cost and measured outcomes.")


def main():
    data = load_data()
    _, metadata = load_model()
    with st.sidebar:
        st.markdown(f"<h2 style='color:{NAVY}'>NYANJA COMMERCIAL BANK</h2>", unsafe_allow_html=True)
        st.caption("Customer intelligence & retention")
        page = st.radio("Workspace", ["Executive Overview", "Customer Intelligence",
                                      "Retention Prioritisation", "Prediction Lab", "Model Governance"])
        st.divider()
        st.caption("Monetary values are presented in MWK.")
    if page == "Executive Overview":
        overview(data, metadata)
    elif page == "Customer Intelligence":
        customer_intelligence(data)
    elif page == "Retention Prioritisation":
        retention_prioritisation(data)
    elif page == "Prediction Lab":
        prediction_lab()
    else:
        model_governance(metadata)
    st.divider()
    st.caption("Banking analytics portfolio • Decision-support outputs require human review.")


if __name__ == "__main__":
    main()
