"""Premium interactive banking customer retention command centre."""
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
from src.retention import RetentionRecommender, allocate_relationship_managers
from src.case_management import build_starter_case_register, validate_case_register, add_case_monitoring_fields, CASE_STATUSES


st.set_page_config(
    page_title="Nyanja | Customer Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#142B4A"
BLUE = "#2F6B9A"
TEAL = "#2B8C82"
GOLD = "#D7A84B"
RED = "#C85D5D"
MUTED = "#64748B"


def apply_theme():
    """Apply a restrained, consistent banking-product visual system."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root { color-scheme: light !important; }
        html, body, [data-testid="stAppViewContainer"], .stApp,
        [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
            background: #F3F6FA !important; color: #20334A !important;
            font-family: 'DM Sans', sans-serif !important;
        }
        [data-testid="stHeader"] { background: #F3F6FA !important; }
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: #FFFFFF !important; border-right: 1px solid #DCE5EF !important; }
        [data-testid="stSidebar"] * { color: #20334A !important; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: 9px 11px; border-radius: 8px; color: #20334A !important; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #EAF2F8 !important; }
        .block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1500px; }
        .stApp p, .stApp label, .stApp li, .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li,
        .stApp [data-testid="stCaptionContainer"] { color: #263B53 !important; }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp [data-testid="stMarkdownContainer"] h1, .stApp [data-testid="stMarkdownContainer"] h2,
        .stApp [data-testid="stMarkdownContainer"] h3, .stApp [data-testid="stMarkdownContainer"] h4 {
            color: #142B4A !important; font-family: 'Manrope', sans-serif !important;
            letter-spacing: -0.025em; font-weight: 800 !important;
        }
        .stApp [data-testid="stWidgetLabel"] *, .stApp [data-testid="stMetricLabel"] *,
        .stApp [data-testid="stMetricValue"] * { color: #142B4A !important; }
        .stApp input, .stApp textarea, .stApp [data-baseweb="select"] *, .stApp [data-baseweb="input"] * {
            color: #20334A !important; background-color: #FFFFFF !important;
        }
        .stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"] > div {
            background: #FFFFFF !important; border-color: #C9D5E2 !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { color: #142B4A !important; }
        [data-testid="stMetric"] {
            background: #FFFFFF !important; border: 1px solid #DCE5EF !important;
            padding: 17px 19px; border-radius: 13px; box-shadow: 0 2px 8px rgba(20,43,74,.055);
        }
        [data-testid="stMetricLabel"] { color: #52657B !important; font-size: .82rem; }
        [data-testid="stMetricValue"] { color: #142B4A !important; font-family: 'Manrope', sans-serif; }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background: #FFFFFF !important; border: 1px solid #DCE5EF; border-radius: 12px; overflow: hidden;
        }
        [data-testid="stAlert"] { border-radius: 10px; }
        .hero {
            background: linear-gradient(115deg, #E7F0F8 0%, #F5F8FC 62%, #E4F2F0 100%);
            padding: 28px 32px; border-radius: 18px; color: #142B4A; margin-bottom: 22px;
            border: 1px solid #DCE5EF; box-shadow: 0 8px 22px rgba(20,43,74,.06);
        }
        .hero h1 { color: #142B4A !important; font-size: 2rem; margin: 0 0 7px 0; }
        .hero p { color: #52657B !important; margin: 0; font-size: .98rem; }
        .eyebrow { color: #2F6B9A; text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 700; margin-bottom: 9px; }
        .section-kicker { color: #24658E !important; text-transform: uppercase; letter-spacing: .12em; font-size: .7rem; font-weight: 700; margin-bottom: 4px; }
        .insight { background: #FFFFFF; border: 1px solid #DCE5EF; border-left: 4px solid #23877D; border-radius: 10px; padding: 14px 16px; min-height: 92px; }
        .insight strong { color: #142B4A !important; display: block; margin-bottom: 4px; }
        .insight span { color: #52657B !important; font-size: .88rem; }
        .risk-pill { display: inline-block; padding: 4px 9px; border-radius: 99px; font-size: .75rem; font-weight: 700; }
        .footer { color: #66788D !important; font-size: .78rem; border-top: 1px solid #DCE5EF; padding-top: 15px; margin-top: 28px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    path = ROOT / "models" / "logistic_regression_pipeline.joblib"
    if not path.exists():
        return None, {}
    payload = joblib.load(path)
    return ChurnModelPipeline.load(path), payload.get("metadata", {})


@st.cache_data
def load_data():
    score_file = ROOT / "data" / "processed" / "churn_customer_scores.csv"
    test_file = ROOT / "data" / "processed" / "churn_predictions.csv"
    customer_file = ROOT / "data" / "processed" / "churn_customers_clean.csv"
    if not customer_file.exists():
        return None
    customers = pd.read_csv(customer_file).drop_duplicates(subset=["Customer_ID"], keep="first")
    if score_file.exists():
        scores = pd.read_csv(score_file)
        score_scope = "Full population"
    else:
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
    data = scores.merge(customers, on="Customer_ID", how="left", suffixes=("", "_profile"))
    data["Score_Scope"] = score_scope
    if "Actual_Churn" not in data and "Churn" in data:
        data["Actual_Churn"] = data["Churn"]
    if "Risk_Band" not in data:
        data["Risk_Band"] = data["Churn_Probability"].apply(
            RetentionRecommender().assign_risk_band
        )
    return data


def money(value):
    return f"MWK {value:,.0f}"


def page_header(kicker, title, subtitle):
    """Render a clean light-theme page banner with strong text contrast."""
    st.markdown(
        f'<div class="hero" style="background:#EAF2F8 !important;'
        f'background-image:linear-gradient(115deg,#E7F0F8 0%,#F7F9FC 62%,#E6F3F0 100%) !important;'
        f'color:#142B4A !important;padding:30px 34px;border-radius:18px;'
        f'margin:0 0 24px 0;border:1px solid #DCE5EF;box-shadow:0 8px 22px rgba(20,43,74,.06);">'
        f'<div class="eyebrow" style="color:#2F6B9A !important;'
        f'font-size:.76rem;font-weight:800;letter-spacing:.13em;'
        f'text-transform:uppercase;margin:0 0 11px 0;">{kicker}</div>'
        f'<h1 style="display:block !important;color:#142B4A !important;'
        f'-webkit-text-fill-color:#142B4A !important;font-family:Arial,sans-serif !important;'
        f'font-size:clamp(1.8rem,3vw,2.55rem) !important;line-height:1.18 !important;'
        f'font-weight:800 !important;letter-spacing:-.025em !important;'
        f'margin:0 0 10px 0 !important;">{title}</h1>'
        f'<p style="display:block !important;color:#52657B !important;'
        f'-webkit-text-fill-color:#52657B !important;font-family:Arial,sans-serif !important;'
        f'font-size:1.05rem !important;line-height:1.55 !important;'
        f'font-weight:500 !important;margin:0 !important;max-width:900px;">{subtitle}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

def section(title, note=None):
    st.markdown(f'<div class="section-kicker">Portfolio intelligence</div><h3>{title}</h3>',
                unsafe_allow_html=True)
    if note:
        st.caption(note)


def chart_bar(series, xlabel=None, ylabel=None, horizontal=False, percent=False):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    values = series.sort_values(ascending=horizontal)
    labels = [str(x) for x in values.index]
    numbers = values.values * 100 if percent else values.values
    if horizontal:
        bars = ax.barh(labels, numbers, color=BLUE, height=0.62)
        ax.invert_yaxis()
        for bar, value in zip(bars, numbers):
            ax.text(value + max(numbers.max() * .015, .2), bar.get_y() + bar.get_height()/2,
                    f"{value:.1f}%" if percent else f"{value:,.0f}", va="center", fontsize=8, color=NAVY)
    else:
        bars = ax.bar(labels, numbers, color=TEAL, width=.62)
        for bar, value in zip(bars, numbers):
            ax.text(bar.get_x() + bar.get_width()/2, value + max(numbers.max() * .015, .2),
                    f"{value:.1f}%" if percent else f"{value:,.0f}", ha="center", fontsize=8, color=NAVY)
    ax.set_ylabel(ylabel or "")
    ax.set_xlabel(xlabel or "")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#DCE4ED")
    ax.tick_params(axis="both", labelsize=8, colors=MUTED, length=0)
    ax.grid(axis="x" if horizontal else "y", alpha=.18, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def risk_mix(data):
    counts = data["Risk_Band"].value_counts().reindex(
        ["Low Risk", "Medium Risk", "High Risk"]
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(5.8, 3.25))
    colors = ["#5C9E83", GOLD, RED]
    wedges, _, _ = ax.pie(
        counts.values, labels=None, autopct="%1.1f%%", startangle=90,
        colors=colors, pctdistance=.78,
        wedgeprops={"width": .34, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 9, "color": NAVY},
    )
    ax.text(0, .08, f"{len(data):,}", ha="center", va="center",
            fontsize=21, fontweight="bold", color=NAVY)
    ax.text(0, -.18, "CUSTOMERS", ha="center", va="center",
            fontsize=7, color=MUTED, fontweight="bold")
    ax.legend(wedges, [f"{k}  ·  {int(v):,}" for k, v in counts.items()],
              loc="lower center", bbox_to_anchor=(.5, -.12), ncol=3,
              frameon=False, fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def overview(data, metadata):
    page_header("Nyanja Commercial Bank  /  Intelligence Suite",
                "Customer Retention Command Centre",
                "A decision workspace for portfolio health, emerging churn exposure and targeted retention.")
    if data is None or data.empty:
        st.warning("Analytics exports are missing. Run scripts/train_pipeline.py from the repository root.")
        return

    score_scope = data["Score_Scope"].iloc[0]
    if not score_scope.startswith("Full population"):
        st.warning("This workspace is currently using the held-out test cohort, not the full customer population.")
    else:
        st.caption("Portfolio view: full-population scores  ·  Model validation: independent held-out test set")

    total = len(data)
    churn_rate = data["Actual_Churn"].mean()
    high = data[data["Risk_Band"] == "High Risk"]
    medium = data[data["Risk_Band"] == "Medium Risk"]
    high_balance = high["Average_Monthly_Balance"].sum() if "Average_Monthly_Balance" in data else 0
    avg_probability = data["Churn_Probability"].mean()

    metrics = st.columns(5)
    metrics[0].metric("Customers monitored", f"{total:,}")
    metrics[1].metric("Historical churn", f"{churn_rate:.1%}")
    metrics[2].metric("High-risk customers", f"{len(high):,}", f"{len(high)/total:.1%} of portfolio")
    metrics[3].metric("High-risk balances", money(high_balance))
    metrics[4].metric("Average risk score", f"{avg_probability:.1%}")

    st.markdown("")
    left, right = st.columns([1.12, .88], gap="large")
    with left:
        section("Where churn is concentrated", "Observed churn rate by customer segment.")
        if "Customer_Segment" in data:
            chart_bar(data.groupby("Customer_Segment", observed=True)["Actual_Churn"].mean(),
                      horizontal=True, percent=True)
    with right:
        section("Portfolio risk profile", "Share of customers in each model risk band.")
        risk_mix(data)

    left, right = st.columns(2, gap="large")
    with left:
        section("Branch performance", "Branches ranked by observed churn rate.")
        if "Branch_Location" in data:
            branch = data.groupby("Branch_Location", observed=True)["Actual_Churn"].mean().sort_values(ascending=False).head(8)
            chart_bar(branch, horizontal=True, percent=True)
    with right:
        section("Inactivity and attrition", "Historical churn by days since last transaction.")
        if "Days_Since_Last_Transaction" in data:
            bins = pd.cut(data["Days_Since_Last_Transaction"],
                          [-1, 30, 60, 90, 180, float("inf")],
                          labels=["0–30", "31–60", "61–90", "91–180", "180+"])
            activity = data.assign(Inactivity_Band=bins).groupby(
                "Inactivity_Band", observed=True
            )["Actual_Churn"].mean()
            chart_bar(activity, percent=True)

    st.markdown("")
    section("Priority watchlist", "Highest-scoring customers for relationship-team review.")
    watch = data.sort_values("Churn_Probability", ascending=False).head(8)
    fields = [c for c in ["Customer_ID", "Customer_Segment", "Branch_Location",
                           "Churn_Probability", "Risk_Band", "Days_Since_Last_Transaction",
                           "Number_of_Complaints"] if c in watch]
    st.dataframe(watch[fields].style.format({"Churn_Probability": "{:.1%}"}),
                 use_container_width=True, hide_index=True)

    metrics = metadata.get("test_metrics", {})
    if metrics:
        st.markdown("")
        section("Model validation snapshot", "Independent performance on the held-out test set.")
        cols = st.columns(5)
        for col, key, label in zip(
            cols, ["roc_auc", "recall", "precision", "f1", "balanced_accuracy"],
            ["ROC-AUC", "Recall", "Precision", "F1 score", "Balanced accuracy"],
        ):
            value = metrics.get(key)
            col.metric(label, f"{value:.1%}" if isinstance(value, (int, float)) else "—")


def customer_intelligence(data):
    page_header("Explore  /  Segment  /  Understand",
                "Customer Intelligence",
                "Slice the portfolio by branch, segment and risk to uncover behavioural patterns.")
    if data is None:
        st.warning("Run the training pipeline to generate the customer analysis exports.")
        return

    c = st.columns(4)
    segments = sorted(data["Customer_Segment"].dropna().unique()) if "Customer_Segment" in data else []
    branches = sorted(data["Branch_Location"].dropna().unique()) if "Branch_Location" in data else []
    chosen_segments = c[0].multiselect("Customer segment", segments, default=segments)
    chosen_branches = c[1].multiselect("Branch", branches, default=branches)
    chosen_risk = c[2].multiselect("Risk band", ["Low Risk", "Medium Risk", "High Risk"],
                                   default=["Low Risk", "Medium Risk", "High Risk"])
    min_probability = c[3].slider("Minimum risk score", 0, 100, 0, 5) / 100

    view = data.copy()
    if "Customer_Segment" in view:
        view = view[view["Customer_Segment"].isin(chosen_segments)]
    if "Branch_Location" in view:
        view = view[view["Branch_Location"].isin(chosen_branches)]
    view = view[view["Risk_Band"].isin(chosen_risk)]
    view = view[view["Churn_Probability"] >= min_probability]

    st.metric("Customers in current view", f"{len(view):,}")
    if view.empty:
        st.info("No customers match these filters. Broaden the selections to see the portfolio.")
        return

    left, right = st.columns(2, gap="large")
    with left:
        section("Complaints and churn")
        if "Number_of_Complaints" in view:
            temp = view.assign(Complaint_Group=view["Number_of_Complaints"].clip(upper=4))
            chart_bar(temp.groupby("Complaint_Group", observed=True)["Actual_Churn"].mean(),
                      percent=True)
    with right:
        section("Product depth and churn")
        if "Number_of_Products" in view:
            chart_bar(view.groupby("Number_of_Products", observed=True)["Actual_Churn"].mean(),
                      percent=True)

    section("Customer-level portfolio explorer", "Sort by churn score and inspect the profile indicators.")
    fields = [x for x in ["Customer_ID", "Customer_Segment", "Branch_Location",
        "Churn_Probability", "Risk_Band", "Days_Since_Last_Transaction",
        "Number_of_Complaints", "Number_of_Products", "Digital_Engagement_Score",
        "Average_Monthly_Balance"] if x in view]
    st.dataframe(view[fields].sort_values("Churn_Probability", ascending=False).style.format(
        {"Churn_Probability": "{:.1%}", "Average_Monthly_Balance": "MWK {:,.0f}"}),
        use_container_width=True, hide_index=True)
    st.download_button("Export filtered customer view",
        view[fields].to_csv(index=False).encode("utf-8"),
        "banking_customer_intelligence.csv", "text/csv")

    st.markdown("")
    section("Customer 360° profile", "Select an individual record for a consolidated relationship-team review.")
    customer_ids = view["Customer_ID"].dropna().astype(str).sort_values().unique().tolist()
    if customer_ids:
        selected_id = st.selectbox("Select customer", customer_ids, key="customer_360_id")
        profile = view[view["Customer_ID"].astype(str) == selected_id].iloc[0]
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Churn probability", f"{profile['Churn_Probability']:.1%}")
        p2.metric("Risk classification", str(profile["Risk_Band"]))
        if "Average_Monthly_Balance" in profile.index:
            p3.metric("Average monthly balance", money(profile["Average_Monthly_Balance"]))
        if "Days_Since_Last_Transaction" in profile.index:
            p4.metric("Days inactive", f"{profile['Days_Since_Last_Transaction']:,.0f}")
        profile_fields = [
            ("Customer segment", "Customer_Segment"), ("Branch", "Branch_Location"),
            ("Tenure (months)", "Account_Tenure_Months"), ("Products held", "Number_of_Products"),
            ("Complaints", "Number_of_Complaints"), ("Digital engagement", "Digital_Engagement_Score"),
            ("Credit score", "Credit_Score"), ("Age", "Age"),
        ]
        profile_rows = [
            {"Profile attribute": label, "Value": profile[column]}
            for label, column in profile_fields
            if column in profile.index and pd.notna(profile[column])
        ]
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)
        recommendation = RetentionRecommender().recommend_action(
            profile, float(profile["Churn_Probability"]), str(profile["Risk_Band"])
        )
        reason_text = "; ".join(recommendation["reason_codes"]) or "No rule-based reason flags were triggered."
        st.markdown(
            f'<div class="insight"><strong>Suggested relationship-team action · '
            f'{recommendation["priority"]}</strong><span>{recommendation["recommended_action"]}'
            f'<br/>Review indicators: {reason_text}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No customer profiles are available for the current filters.")


def retention_prioritisation(data):
    page_header("Activate  /  Prioritise  /  Retain",
                "Retention Prioritisation",
                "Translate model scores into a capacity-aware action list for relationship teams.")
    if data is None:
        st.warning("Prediction exports are missing. Run the training pipeline first.")
        return
    if not data["Score_Scope"].iloc[0].startswith("Full population"):
        st.info("The queue currently covers only the held-out test cohort. Refresh the deployment after the full-population scoring update.")

    a, b, c = st.columns([1, 1, 2])
    capacity = a.slider("Team capacity", 25, 2000, 500, 25)
    minimum_band = b.selectbox("Include customers", ["High and medium risk", "High risk only"])
    include_medium = minimum_band == "High and medium risk"
    threshold = c.slider("Minimum probability", 0, 100, 33, 1) / 100

    eligible = data[data["Risk_Band"].eq("High Risk") |
                    (data["Risk_Band"].eq("Medium Risk") if include_medium else False)]
    eligible = eligible[eligible["Churn_Probability"] >= threshold]
    queue = RetentionRecommender().create_outreach_queue(eligible, capacity=capacity)
    if queue.empty:
        st.info("No customers meet the selected risk and capacity criteria.")
        return

    critical = int((queue["priority"] == "Critical").sum())
    high = int((queue["priority"] == "High").sum())
    avg_score = queue["Churn_Probability"].mean()
    k = st.columns(4)
    k[0].metric("Recommended contacts", f"{len(queue):,}")
    k[1].metric("Critical priority", f"{critical:,}")
    k[2].metric("High priority", f"{high:,}")
    k[3].metric("Average queue risk", f"{avg_score:.1%}")

    left, right = st.columns([1, 1], gap="large")
    with left:
        section("Queue risk composition")
        risk_mix(queue)
    with right:
        section("Priority mix")
        priority_counts = queue["priority"].value_counts().reindex(
            ["Critical", "High", "Medium", "Low", "None"]
        ).dropna()
        chart_bar(priority_counts, horizontal=True)

    section("Relationship-team worklist", "Ranked by score. Recommendations are prompts for human review.")
    fields = [x for x in ["Outreach_Rank", "Customer_ID", "Risk_Band",
        "Churn_Probability", "priority", "recommended_action", "Days_Since_Last_Transaction",
        "Number_of_Complaints", "Customer_Segment", "Branch_Location",
        "Average_Monthly_Balance"] if x in queue]
    st.dataframe(queue[fields].style.format(
        {"Churn_Probability": "{:.1%}", "Average_Monthly_Balance": "MWK {:,.0f}"}),
        use_container_width=True, hide_index=True)
    st.download_button("Download relationship-team worklist",
        queue.to_csv(index=False).encode("utf-8"),
        "nyanja_retention_worklist.csv", "text/csv")

    st.markdown("")
    section("Campaign budget & outcome planner",
            "Translate the ranked queue into a budget scenario. Response assumptions are user inputs, not model predictions.")
    budget_col, cost_col, response_col = st.columns(3)
    campaign_budget = budget_col.number_input(
        "Campaign budget (MWK)", min_value=0, max_value=100_000_000,
        value=2_500_000, step=100_000, format="%d"
    )
    cost_per_contact = cost_col.number_input(
        "Estimated cost per contact (MWK)", min_value=1, max_value=1_000_000,
        value=5_000, step=500, format="%d"
    )
    response_rate = response_col.slider(
        "Assumed successful retention rate", min_value=1, max_value=50,
        value=15, step=1
    ) / 100

    budget_capacity = int(campaign_budget // cost_per_contact)
    planned_contacts = min(len(queue), capacity, budget_capacity)
    planned_queue = queue.head(planned_contacts).copy()
    estimated_retained = int(round(planned_contacts * response_rate))
    planned_spend = planned_contacts * cost_per_contact
    average_balance = (
        planned_queue["Average_Monthly_Balance"].sum()
        if planned_contacts and "Average_Monthly_Balance" in planned_queue else 0
    )
    p = st.columns(4)
    p[0].metric("Budget-supported contacts", f"{planned_contacts:,}",
                f"{budget_capacity:,} affordable at this unit cost")
    p[1].metric("Planned campaign spend", money(planned_spend))
    p[2].metric("Scenario retained customers", f"{estimated_retained:,}",
                f"{response_rate:.0%} assumed response")
    cost_per_save = planned_spend / estimated_retained if estimated_retained else None
    p[3].metric("Scenario cost per retained", money(cost_per_save) if cost_per_save else "—")

    st.caption(
        f"Selected {planned_contacts:,} highest-priority customers from the queue. "
        f"Balances associated with this contact group: {money(average_balance)}. "
        "This balance is exposure context, not revenue, profit or guaranteed savings."
    )
    if planned_contacts:
        st.download_button(
            "Download budget-aligned campaign list",
            planned_queue.to_csv(index=False).encode("utf-8"),
            "nyanja_budget_aligned_campaign.csv",
            "text/csv",
        )
    else:
        st.info("Increase the campaign budget or reduce the estimated cost per contact to fund outreach.")


def campaign_studio(data):
    page_header(
        "Plan  /  Assign  /  Activate",
        "Retention Campaign Studio",
        "Build a targeted outreach campaign, balance work across relationship teams and export an execution-ready contact list.",
    )
    if data is None or data.empty:
        st.warning("Full customer scoring data is unavailable. Run the production scoring pipeline first.")
        return

    st.caption(
        "Campaigns created here are planning exports. Assignments are not saved to a live CRM or "
        "customer system; upload actual contact outcomes in Intervention Outcomes after outreach."
    )
    left, right = st.columns(2)
    campaign_name = left.text_input("Campaign name", value="Priority retention outreach")
    channel = right.selectbox(
        "Primary contact channel",
        ["Relationship manager call", "Branch appointment", "SMS", "Email", "Digital banking message"],
    )
    segments = sorted(data["Customer_Segment"].dropna().astype(str).unique()) if "Customer_Segment" in data else []
    branches = sorted(data["Branch_Location"].dropna().astype(str).unique()) if "Branch_Location" in data else []
    c1, c2, c3 = st.columns(3)
    selected_segments = c1.multiselect("Target customer segments", segments, default=segments)
    selected_branches = c2.multiselect("Target branches", branches, default=branches)
    risk_options = ["High Risk", "Medium Risk", "Low Risk"]
    selected_risks = c3.multiselect("Risk bands", risk_options, default=["High Risk", "Medium Risk"])

    c4, c5, c6 = st.columns(3)
    minimum_score = c4.slider("Minimum churn score", 0, 100, 33, 1) / 100
    capacity = c5.number_input("Maximum campaign contacts", min_value=1, max_value=20000,
                                value=min(500, len(data)), step=25)
    contact_date = c6.date_input("Planned contact date")

    managers_raw = st.text_input(
        "Relationship managers / teams (comma-separated)",
        value="Team A, Team B, Team C",
        help="Replace these labels with the actual authorised team or manager names before exporting.",
    )
    managers = list(dict.fromkeys(name.strip() for name in managers_raw.split(",") if name.strip()))

    eligible = data.copy()
    if "Customer_Segment" in eligible:
        eligible = eligible[eligible["Customer_Segment"].astype(str).isin(selected_segments)]
    if "Branch_Location" in eligible:
        eligible = eligible[eligible["Branch_Location"].astype(str).isin(selected_branches)]
    eligible = eligible[
        eligible["Risk_Band"].isin(selected_risks)
        & (eligible["Churn_Probability"] >= minimum_score)
    ]
    if not data["Score_Scope"].iloc[0].startswith("Full population"):
        st.warning("The source scores cover only a held-out test cohort. Do not use this as a full-population campaign list.")

    st.metric("Eligible customers", f"{len(eligible):,}")
    if not campaign_name.strip():
        st.error("Enter a campaign name before creating the export.")
        return
    if not managers:
        st.error("Enter at least one relationship manager or team.")
        return
    if eligible.empty:
        st.info("No customers match the selected campaign criteria.")
        return

    queue = RetentionRecommender().create_outreach_queue(eligible, capacity=int(capacity))
    queue = allocate_relationship_managers(queue, managers)
    queue["Campaign_Name"] = campaign_name.strip()
    queue["Contact_Channel"] = channel
    queue["Planned_Contact_Date"] = str(contact_date)

    summary = queue.groupby("Assigned_Manager").agg(
        Assigned_Customers=("Customer_ID", "nunique"),
        Average_Churn_Score=("Churn_Probability", "mean"),
        High_Risk_Customers=("Risk_Band", lambda values: int((values == "High Risk").sum())),
    ).reset_index()
    k = st.columns(4)
    k[0].metric("Campaign contacts", f"{len(queue):,}")
    k[1].metric("Managers / teams", f"{len(managers):,}")
    k[2].metric("Highest individual workload", f"{summary['Assigned_Customers'].max():,}")
    k[3].metric("Average queue score", f"{queue['Churn_Probability'].mean():.1%}")

    section("Balanced manager workload", "Customers are allocated in ranked order, with workloads differing by at most one.")
    st.dataframe(summary.style.format({"Average_Churn_Score": "{:.1%}"}),
                 use_container_width=True, hide_index=True)
    section("Campaign execution list", "Review the proposed assignments before sharing with authorised teams.")
    display = [column for column in [
        "Outreach_Rank", "Customer_ID", "Assigned_Manager", "Campaign_Name",
        "Contact_Channel", "Planned_Contact_Date", "Risk_Band", "Churn_Probability",
        "priority", "recommended_action", "Customer_Segment", "Branch_Location",
        "Days_Since_Last_Transaction", "Number_of_Complaints", "Average_Monthly_Balance",
    ] if column in queue.columns]
    st.dataframe(queue[display].style.format({
        "Churn_Probability": "{:.1%}", "Average_Monthly_Balance": "MWK {:,.0f}"
    }), use_container_width=True, hide_index=True)
    st.download_button(
        "Export campaign execution list",
        queue.to_csv(index=False).encode("utf-8"),
        "nyanja_retention_campaign_execution.csv",
        "text/csv",
    )
    st.caption("The export contains customer identifiers and should be handled as confidential banking information.")



def retention_case_management(data):
    page_header(
        "Own  /  Follow up  /  Resolve",
        "Retention Case Management",
        "Turn risk signals into accountable customer cases, with clear ownership, next actions and follow-up control.",
    )
    if data is None or data.empty:
        st.warning("Customer scoring data is unavailable. Run the production scoring pipeline first.")
        return
    st.info(
        "Working register: import the latest CSV to resume work, edit cases below, then export the updated CSV. "
        "This app does not currently write case changes to a permanent shared database."
    )
    starter = build_starter_case_register(data, limit=100)
    st.download_button(
        "Download starter case register (top 100 priority customers)",
        starter.to_csv(index=False).encode("utf-8"),
        "nyanja_retention_case_register_template.csv",
        "text/csv",
    )
    upload = st.file_uploader("Upload current case register (CSV)", type=["csv"], key="retention_case_register_upload")
    if upload is None:
        st.caption("Download the starter file, complete ownership and dates, then upload it here to manage cases.")
        return
    try:
        raw = pd.read_csv(upload, dtype={"Customer_ID": str})
        cases = validate_case_register(raw)
    except Exception as exc:
        st.error(f"Case register could not be loaded: {exc}")
        return

    portfolio = data.copy()
    portfolio["Customer_ID"] = portfolio["Customer_ID"].astype(str)
    profile_fields = [c for c in [
        "Customer_ID", "Churn_Probability", "Risk_Band", "Customer_Segment",
        "Branch_Location", "Average_Monthly_Balance", "Days_Since_Last_Transaction",
    ] if c in portfolio.columns]
    cases = cases.merge(portfolio[profile_fields], on="Customer_ID", how="left", validate="many_to_one")
    cases = add_case_monitoring_fields(cases)
    k = st.columns(5)
    k[0].metric("Cases in register", f"{len(cases):,}")
    k[1].metric("Open cases", f"{int(cases['Case_Open'].sum()):,}")
    k[2].metric("Overdue follow-ups", f"{int(cases['Follow_Up_Overdue'].sum()):,}")
    k[3].metric("Unassigned cases", f"{int(cases['Assigned_Manager'].eq('').sum()):,}")
    k[4].metric("Not matched to portfolio", f"{int(cases['Churn_Probability'].isna().sum()):,}")

    status_filter = st.multiselect("Filter case status", CASE_STATUSES, default=CASE_STATUSES)
    overdue_only = st.checkbox("Show overdue follow-ups only", value=False)
    view = cases[cases["Case_Status"].isin(status_filter)].copy()
    if overdue_only:
        view = view[view["Follow_Up_Overdue"]]
    if view.empty:
        st.info("No cases match the current filters.")
        return
    section("Case workbench", "Edit case status, owner, follow-up date and latest interaction. Keep Case_ID and Customer_ID unique.")
    editable_cols = [
        "Case_ID", "Customer_ID", "Case_Status", "Assigned_Manager", "Next_Follow_Up",
        "Last_Contact_Date", "Contact_Channel", "Latest_Outcome", "Resolution_Reason", "Case_Notes",
    ]
    edited = st.data_editor(
        view[editable_cols], use_container_width=True, hide_index=True, num_rows="dynamic",
        column_config={
            "Case_Status": st.column_config.SelectboxColumn("Case status", options=CASE_STATUSES, required=True),
            "Next_Follow_Up": st.column_config.DateColumn("Next follow-up", format="YYYY-MM-DD"),
            "Last_Contact_Date": st.column_config.DateColumn("Last contact", format="YYYY-MM-DD"),
        }, key="retention_case_editor",
    )
    try:
        edited = validate_case_register(edited)
        enriched = edited.merge(portfolio[profile_fields], on="Customer_ID", how="left", validate="many_to_one")
        enriched = add_case_monitoring_fields(enriched)
    except Exception as exc:
        st.error(f"Please correct the case register before exporting: {exc}")
        return
    overdue = enriched[enriched["Follow_Up_Overdue"]]
    if not overdue.empty:
        st.warning(f"{len(overdue):,} open case(s) have a follow-up date before today.")
        st.dataframe(overdue[[c for c in ["Case_ID", "Customer_ID", "Assigned_Manager", "Case_Status", "Next_Follow_Up", "Churn_Probability"] if c in overdue]],
                     use_container_width=True, hide_index=True)
    section("Risk context", "Churn scores are prioritisation signals, not confirmed future outcomes.")
    risk_cols = [c for c in ["Case_ID", "Customer_ID", "Case_Status", "Assigned_Manager",
        "Churn_Probability", "Risk_Band", "Branch_Location", "Next_Follow_Up", "Follow_Up_Overdue"] if c in enriched]
    st.dataframe(enriched[risk_cols].sort_values("Churn_Probability", ascending=False, na_position="last").style.format(
        {"Churn_Probability": "{:.1%}"}), use_container_width=True, hide_index=True)
    st.download_button(
        "Export updated case register", enriched.to_csv(index=False).encode("utf-8"),
        "nyanja_retention_case_register_updated.csv", "text/csv",
    )
    st.caption("Case exports contain confidential banking information. Store them only in approved, access-controlled locations.")


def prediction_lab():
    page_header("Scenario testing  /  Batch scoring",
                "Prediction Lab",
                "Review the model's independent validation evidence, then score a new customer file.")
    model, metadata = load_model()
    if model is None:
        st.warning("Trained model not found. Run scripts/train_pipeline.py first.")
        return

    metrics = metadata.get("test_metrics", {})
    if metrics:
        section("Model validation reference", "Final metrics below are from the untouched held-out test partition.")
        validation_cols = st.columns(5)
        for col, key, label in zip(
            validation_cols, ["roc_auc", "average_precision", "recall", "precision", "f1"],
            ["ROC-AUC", "PR-AUC (average precision)", "Recall", "Precision", "F1 score"],
        ):
            value = metrics.get(key)
            col.metric(label, f"{value:.1%}" if isinstance(value, (int, float)) else "—")
        st.caption(
            "These metrics describe historical test-set performance, not guaranteed future accuracy. "
            "Threshold scenarios are kept separate and are calculated on a validation split from training data."
        )

    upload = st.file_uploader("Upload customer data (CSV)", type=["csv"])
    st.caption("Required fields are the same raw customer attributes used during model training.")
    if upload is None:
        st.info("Upload a CSV to begin. Your file is processed within this app session.")
        return
    try:
        customers = pd.read_csv(upload)
        missing = sorted(set(model.feature_names_in_) - set(customers.columns))
        if missing:
            st.error("Missing required fields: " + ", ".join(missing))
            return
        scored = customers.copy()
        scored["Churn_Probability"] = model.predict_proba(customers)[:, 1]
        recommender = RetentionRecommender()
        scored["Risk_Band"] = scored["Churn_Probability"].map(recommender.assign_risk_band)
        scored["Predicted_Churn_0.50"] = (scored["Churn_Probability"] >= .50).astype(int)
        st.success(f"Successfully scored {len(scored):,} customer records.")
        high = int((scored["Risk_Band"] == "High Risk").sum())
        medium = int((scored["Risk_Band"] == "Medium Risk").sum())
        low = int((scored["Risk_Band"] == "Low Risk").sum())
        c = st.columns(4)
        c[0].metric("Records scored", f"{len(scored):,}")
        c[1].metric("High risk", f"{high:,}")
        c[2].metric("Medium risk", f"{medium:,}")
        c[3].metric("Low risk", f"{low:,}")
        section("Scoring distribution")
        risk_mix(scored)
        section("Ranked scoring results")
        st.dataframe(scored.sort_values("Churn_Probability", ascending=False),
                     use_container_width=True, hide_index=True)
        st.download_button("Download scored customer file",
            scored.to_csv(index=False).encode("utf-8"),
            "nyanja_customer_churn_scores.csv", "text/csv")
    except Exception as exc:
        st.error(f"Unable to score this file: {exc}")


def intervention_outcomes(data):
    page_header(
        "Measure  /  Learn  /  Improve",
        "Intervention Outcomes",
        "Review completed outreach and compare observed outcomes across campaigns, branches and risk groups.",
    )
    if data is None:
        st.warning("Customer scoring data is unavailable. Run the training pipeline first.")
        return
    st.markdown(
        "Upload a relationship-team outcome file with one row per customer. Required columns: "
        "`Customer_ID`, `Contacted` (1/0), and `Retained` (1/0). Optional columns: "
        "`Campaign`, `Contact_Date`, and `Outcome_Notes`."
    )
    template = pd.DataFrame({
        "Customer_ID": ["CUST-00001"], "Contacted": [1], "Retained": [1],
        "Campaign": ["Win-back pilot"], "Contact_Date": ["2026-11-01"],
        "Outcome_Notes": ["Customer agreed to review product options"],
    })
    st.download_button("Download outcome-file template",
        template.to_csv(index=False).encode("utf-8"),
        "nyanja_intervention_outcomes_template.csv", "text/csv")
    upload = st.file_uploader("Upload intervention outcomes (CSV)", type=["csv"], key="outcomes_upload")
    if upload is None:
        st.info("Upload an outcomes file to review contact coverage and observed retention results.")
        return
    try:
        outcomes = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        return
    required = {"Customer_ID", "Contacted", "Retained"}
    missing = sorted(required - set(outcomes.columns))
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return
    outcomes = outcomes.copy()
    for column in ["Contacted", "Retained"]:
        outcomes[column] = pd.to_numeric(outcomes[column], errors="coerce")
    if outcomes[["Contacted", "Retained"]].isna().any().any():
        st.error("Contacted and Retained must contain only 1 or 0 values.")
        return
    if not outcomes["Contacted"].isin([0, 1]).all() or not outcomes["Retained"].isin([0, 1]).all():
        st.error("Contacted and Retained must contain only 1 or 0 values.")
        return
    if (outcomes["Retained"] > outcomes["Contacted"]).any():
        st.error("A customer cannot be marked retained when Contacted is 0.")
        return
    outcomes["Customer_ID"] = outcomes["Customer_ID"].astype(str)
    outcomes = outcomes.drop_duplicates("Customer_ID", keep="last")
    scored = data.copy()
    scored["Customer_ID"] = scored["Customer_ID"].astype(str)
    report = outcomes.merge(
        scored[["Customer_ID", "Churn_Probability", "Risk_Band", "Branch_Location"]],
        on="Customer_ID", how="left", suffixes=("", "_portfolio"),
    )
    contacted = report[report["Contacted"] == 1]
    matched = contacted["Churn_Probability"].notna().sum()
    retention_rate = contacted["Retained"].mean() if len(contacted) else 0
    k = st.columns(4)
    k[0].metric("Outcome records", f"{len(report):,}")
    k[1].metric("Customers contacted", f"{len(contacted):,}")
    k[2].metric("Matched to scored portfolio", f"{matched:,}",
                f"{matched / len(contacted):.1%} of contacted" if len(contacted) else "No contacts")
    k[3].metric("Observed retention rate", f"{retention_rate:.1%}", "Among contacted records")
    if "Campaign" in report.columns:
        section("Campaign-level outcomes", "Descriptive results; comparisons do not establish causal impact.")
        campaign = contacted.groupby("Campaign", dropna=False).agg(
            Customers_Contacted=("Customer_ID", "nunique"),
            Customers_Retained=("Retained", "sum"),
            Retention_Rate=("Retained", "mean"),
        ).reset_index()
        st.dataframe(campaign.style.format({"Retention_Rate": "{:.1%}"}),
                     use_container_width=True, hide_index=True)
    if "Branch_Location" in contacted.columns and contacted["Branch_Location"].notna().any():
        section("Branch-level intervention outcomes")
        branch = contacted.groupby("Branch_Location", dropna=False).agg(
            Customers_Contacted=("Customer_ID", "nunique"),
            Customers_Retained=("Retained", "sum"),
            Retention_Rate=("Retained", "mean"),
        ).reset_index().sort_values("Retention_Rate", ascending=False)
        st.dataframe(branch.style.format({"Retention_Rate": "{:.1%}"}),
                     use_container_width=True, hide_index=True)
    section("Customer outcome register")
    display = [c for c in [
        "Customer_ID", "Campaign", "Contact_Date", "Contacted", "Retained",
        "Churn_Probability", "Risk_Band", "Branch_Location", "Outcome_Notes",
    ] if c in report.columns]
    st.dataframe(report[display].style.format({"Churn_Probability": "{:.1%}"}),
                 use_container_width=True, hide_index=True)
    st.download_button("Download reviewed outcome register",
        report.to_csv(index=False).encode("utf-8"),
        "nyanja_reviewed_intervention_outcomes.csv", "text/csv")
    st.caption(
        "These are observed post-contact outcomes, not proof that an intervention caused retention. "
        "Use an appropriate control group or experiment to estimate incremental impact."
    )


def model_governance(metadata):
    page_header("Validate  /  Monitor  /  Govern",
                "Model Performance & Governance",
                "Understand model discrimination, error trade-offs and the controls required for responsible use.")
    if not metadata:
        st.warning("Model metadata is unavailable. Train the model to generate an evaluation record.")
        return
    metrics = metadata.get("test_metrics", {})
    primary = st.columns(4)
    for col, key, label in zip(primary, ["roc_auc", "recall", "precision", "f1"],
                               ["ROC-AUC", "Recall", "Precision", "F1 score"]):
        value = metrics.get(key)
        col.metric(label, f"{value:.1%}" if isinstance(value, (int, float)) else "—")

    selected_model = metadata.get("selected_model", "Selected candidate")
    section("Held-out model scorecard", f"{selected_model.replace('_', ' ').title()} evaluated on the independent test split.")
    labels = {
        "roc_auc": "ROC-AUC", "average_precision": "Average precision",
        "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
        "precision": "Precision", "recall": "Recall", "f1": "F1 score",
        "specificity": "Specificity", "brier_score": "Brier score", "log_loss": "Log loss",
    }
    metric_frame = pd.DataFrame([
        {"Metric": label, "Value": metrics.get(key)}
        for key, label in labels.items() if key in metrics
    ])
    st.dataframe(metric_frame.style.format({"Value": "{:.3f}"}),
                 use_container_width=True, hide_index=True)

    comparison = metadata.get("model_comparison", [])
    if comparison:
        section("Candidate model comparison", "All candidates are evaluated on the same validation split from training data; the final test set is reserved for the selected model.")
        comparison_frame = pd.DataFrame(comparison)
        display_metrics = [column for column in [
            "model", "roc_auc", "average_precision", "recall", "precision",
            "f1", "accuracy", "balanced_accuracy", "brier_score", "log_loss",
        ] if column in comparison_frame]
        comparison_format = {
            column: "{:.1%}" for column in [
                "roc_auc", "average_precision", "recall", "precision",
                "f1", "accuracy", "balanced_accuracy",
            ] if column in display_metrics
        }
        comparison_format.update({
            column: "{:.3f}" for column in ["brier_score", "log_loss"]
            if column in display_metrics
        })
        st.dataframe(comparison_frame[display_metrics].style.format(comparison_format),
                     use_container_width=True, hide_index=True)
        logistic = comparison_frame[comparison_frame["model"] == "logistic"]
        forest = comparison_frame[comparison_frame["model"] == "random_forest"]
        if not logistic.empty and not forest.empty:
            lr, rf = logistic.iloc[0], forest.iloc[0]
            st.caption(
                f"Logistic Regression recall: {lr['recall']:.1%}; Random Forest recall: {rf['recall']:.1%}. "
                f"Logistic Regression F1: {lr['f1']:.1%}; Random Forest F1: {rf['f1']:.1%}. "
                "Compare average precision and probability-loss metrics alongside recall. "
                "The operating choice depends on missed-churn costs, unnecessary outreach and team capacity."
            )
        rationale = metadata.get("model_selection_rationale")
        if rationale:
            st.info(rationale)

    confusion = metadata.get("test_confusion_matrix")
    if confusion and len(confusion) == 2:
        section("Test-set confusion matrix", "Fixed 0.50 decision threshold. Rows are actual outcomes; columns are model predictions.")
        cm = pd.DataFrame(confusion, index=["Actual: retained", "Actual: churn"], columns=["Predicted: retained", "Predicted: churn"])
        st.dataframe(cm, use_container_width=True)
        counts = metadata.get("test_confusion_counts", {})
        count_cols = st.columns(4)
        for col, key, label in zip(
            count_cols, ["true_negatives", "false_positives", "false_negatives", "true_positives"],
            ["Correctly retained", "False alarms", "Missed churners", "Churners identified"],
        ):
            col.metric(label, f"{counts.get(key, 0):,}")

    pr_curve = metadata.get("pr_curve", {})
    if pr_curve.get("precision") and pr_curve.get("recall"):
        section("Precision–recall curve", "More informative than accuracy when churn is the minority class.")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(pr_curve["recall"], pr_curve["precision"], color=BLUE, linewidth=2.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(alpha=.2, linestyle="--")
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    threshold = metadata.get("threshold_analysis", {})
    if threshold:
        threshold_frame = pd.DataFrame(threshold)
        if {"threshold", "precision", "recall"}.issubset(threshold_frame.columns):
            section("Decision-threshold scenarios", metadata.get(
                "threshold_analysis_source",
                "Threshold scenarios are for decision support; do not optimise on the final test set.",
            ))
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.plot(threshold_frame["threshold"], threshold_frame["precision"],
                    marker="o", linewidth=2, label="Precision", color=BLUE)
            ax.plot(threshold_frame["threshold"], threshold_frame["recall"],
                    marker="o", linewidth=2, label="Recall", color=TEAL)
            ax.plot(threshold_frame["threshold"], threshold_frame["f1"],
                    marker="o", linewidth=2, label="F1 score", color=GOLD)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Probability threshold")
            ax.set_ylabel("Metric")
            ax.grid(alpha=.2, linestyle="--")
            ax.legend(frameon=False)
            ax.spines[["top", "right", "left"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.dataframe(threshold_frame.style.format({
                "threshold": "{:.2f}", "precision": "{:.1%}", "recall": "{:.1%}",
                "f1": "{:.1%}", "share_flagged": "{:.1%}",
            }), use_container_width=True, hide_index=True)

    cv_results = metadata.get("cross_validation", {})
    if cv_results:
        section("Training-only cross-validation", "Five stratified folds on the training partition. The held-out test set is not used here.")
        cv_rows = []
        for metric_name, result in cv_results.items():
            cv_rows.append({
                "Metric": metric_name.replace("_", " ").title(),
                "Mean": result.get("mean"),
                "Standard deviation": result.get("std"),
                "Minimum fold": result.get("min"),
                "Maximum fold": result.get("max"),
            })
        st.dataframe(pd.DataFrame(cv_rows).style.format({
            "Mean": "{:.3f}", "Standard deviation": "{:.3f}",
            "Minimum fold": "{:.3f}", "Maximum fold": "{:.3f}",
        }), use_container_width=True, hide_index=True)

    calibration = metadata.get("calibration_curve", {})
    if calibration.get("prob_true") and calibration.get("prob_pred"):
        section("Probability calibration", "Observed churn frequency versus average predicted probability within fixed-width bins.")
        method = metadata.get("calibration_method")
        if method:
            st.caption(f"Probability calibration: {method.title()} method. " +
                       metadata.get("calibration_training", ""))
        observed = calibration["prob_true"]
        predicted = calibration["prob_pred"]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, label="Perfect calibration")
        ax.plot(predicted, observed, marker="o", color=TEAL, linewidth=2, label="Model")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed churn rate")
        ax.grid(alpha=.2, linestyle="--")
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    subgroup = metadata.get("subgroup_metrics", {})
    for group_column, rows in subgroup.items():
        if rows:
            section(f"Hold-out performance by {group_column.replace('_', ' ').lower()}",
                    "Descriptive comparisons only. Groups with fewer than 50 test customers are suppressed.")
            group_frame = pd.DataFrame(rows)
            st.dataframe(group_frame.style.format({
                "churn_rate": "{:.1%}", "precision": "{:.1%}",
                "recall": "{:.1%}", "f1": "{:.1%}",
            }), use_container_width=True, hide_index=True)

    ece = metadata.get("calibration_ece")
    if ece is not None:
        st.metric("Expected calibration error", f"{ece:.3f}")
        if ece >= 0.10:
            st.warning(
                "Calibration error remains material. Treat scores primarily as a ranking and "
                "prioritisation aid, not as precise individual churn probabilities."
            )

    section("Responsible operating controls")
    controls = [
        ("Human review", "Treat scores as prioritisation signals, not definitive customer outcomes."),
        ("Fairness review", "Compare error rates and intervention outcomes across relevant customer groups."),
        ("Data quality", "Monitor missingness, category drift, outliers and input schema changes."),
        ("Model monitoring", "Track calibration, discrimination and score distribution after each refresh."),
        ("Intervention testing", "Use controlled experiments to establish whether retention actions improve outcomes."),
    ]
    for heading, detail in controls:
        st.markdown(f'<div class="insight"><strong>{heading}</strong><span>{detail}</span></div>',
                    unsafe_allow_html=True)
        st.markdown("")


def main():
    apply_theme()
    data = load_data()
    _, metadata = load_model()
    with st.sidebar:
        st.markdown(
            '<div style="padding:8px 2px 18px 2px">'
            '<div style="font-size:1.8rem">🏦</div>'
            '<div style="font-family:Manrope;font-weight:800;font-size:1.05rem;color:#142B4A">NYANJA COMMERCIAL BANK</div>'
            '<div style="font-size:.76rem;color:#64748B">CUSTOMER INTELLIGENCE SUITE</div></div>',
            unsafe_allow_html=True,
        )
        st.divider()
        page = st.radio(
            "WORKSPACE",
            ["Executive Overview", "Customer Intelligence", "Retention Prioritisation",
             "Intervention Outcomes", "Retention Campaign Studio", "Retention Case Management", "Prediction Lab", "Model Governance"],
            label_visibility="visible",
        )
        st.divider()
        st.caption("PORTFOLIO CONTEXT")
        st.markdown("**Currency**  \nMalawi Kwacha (MWK)")
        st.markdown("**Scoring**  \nFull customer population")
        st.markdown("**Decision use**  \nHuman-reviewed support")
        st.divider()
        st.caption("Workspace build · 2.0")
    if page == "Executive Overview":
        overview(data, metadata)
    elif page == "Customer Intelligence":
        customer_intelligence(data)
    elif page == "Retention Prioritisation":
        retention_prioritisation(data)
    elif page == "Intervention Outcomes":
        intervention_outcomes(data)
    elif page == "Retention Campaign Studio":
        campaign_studio(data)
    elif page == "Retention Case Management":
        retention_case_management(data)
    elif page == "Prediction Lab":
        prediction_lab()
    else:
        model_governance(metadata)
    st.markdown(
        '<div class="footer">NYANJA COMMERCIAL BANK · CUSTOMER ANALYTICS & RETENTION '
        ' <span style="float:right">Decision-support outputs require human review.</span></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
