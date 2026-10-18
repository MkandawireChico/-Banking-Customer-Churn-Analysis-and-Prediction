"""
Streamlit prediction interface for banking customer churn prediction.

This app provides an interactive interface for:
- Single customer churn prediction
- Batch prediction from CSV upload
- Model explainability
- Risk band visualization

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
from src.modeling import ChurnModelPipeline
from src.explainability import explain_prediction
from src.retention import RetentionRecommender

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #555;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model pipeline."""
    model_path = Path("models/logistic_regression_pipeline.joblib")
    if not model_path.exists():
        st.error("Model not found. Please run `python scripts/train_pipeline.py` first.")
        return None
    
    model = ChurnModelPipeline.load(model_path)
    return model


@st.cache_data
def get_feature_defaults():
    """Get default values for input features."""
    return {
        "Age": 37,
        "Gender": "Male",
        "Marital_Status": "Married",
        "Education_Level": "Secondary",
        "Employment_Status": "Employed - Private",
        "Monthly_Income": 438859,
        "Account_Tenure_Months": 40,
        "Customer_Segment": "Retail",
        "Branch_Location": "Blantyre",
        "Number_of_Products": 2,
        "Has_Credit_Card": 0,
        "Has_Mobile_Banking": 1,
        "Has_Internet_Banking": 1,
        "Average_Monthly_Balance": 500000,
        "Average_Monthly_Transactions": 8,
        "Total_Transaction_Value_Last_6_Months": 5000000,
        "Number_of_Complaints": 0,
        "Days_Since_Last_Transaction": 15,
        "Average_Monthly_Deposit": 300000,
        "Average_Monthly_Withdrawal": 250000,
        "Has_Active_Loan": 0,
        "Loan_Status": "No Active Loan",
        "Digital_Engagement_Score": 60,
        "Customer_Service_Interactions": 0,
    }


def single_customer_prediction():
    """Single customer prediction interface."""
    st.markdown('<h1 class="main-header">Single Customer Prediction</h1>', unsafe_allow_html=True)
    
    model = load_model()
    if model is None:
        return
    
    defaults = get_feature_defaults()
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="sub-header">Demographics</h2>', unsafe_allow_html=True)
        age = st.slider("Age", 18, 90, defaults["Age"])
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"], index=1)
        education = st.selectbox("Education Level", ["Primary", "Secondary", "Tertiary/College", "University"], index=1)
        employment = st.selectbox("Employment Status", 
                                ["Employed - Private", "Employed - Government", "Business Owner", 
                                 "Self-Employed", "Student", "Unemployed"], index=0)
        
        st.markdown('<h2 class="sub-header">Account Details</h2>', unsafe_allow_html=True)
        segment = st.selectbox("Customer Segment", ["Retail", "SME", "Premium", "Salary Account", "Student/Youth"], index=0)
        branch = st.selectbox("Branch Location", 
                             ["Blantyre", "Lilongwe", "Mzuzu", "Zomba", "Mangochi",
                              "Kasungu", "Salima", "Liwonde", "Rumphi", "Karonga"], index=0)
        tenure = st.slider("Account Tenure (months)", 1, 240, defaults["Account_Tenure_Months"])
        products = st.slider("Number of Products", 1, 6, defaults["Number_of_Products"])
        income = st.number_input("Monthly Income (MWK)", min_value=50000, max_value=5000000, value=defaults["Monthly_Income"])
    
    with col2:
        st.markdown('<h2 class="sub-header">Digital & Transaction Behavior</h2>', unsafe_allow_html=True)
        has_mobile = st.checkbox("Has Mobile Banking", value=True)
        has_internet = st.checkbox("Has Internet Banking", value=True)
        has_credit = st.checkbox("Has Credit Card", value=False)
        
        digital_score = st.slider("Digital Engagement Score", 0, 100, defaults["Digital_Engagement_Score"])
        avg_transactions = st.slider("Average Monthly Transactions", 0, 60, defaults["Average_Monthly_Transactions"])
        days_inactive = st.slider("Days Since Last Transaction", 0, 400, defaults["Days_Since_Last_Transaction"])
        complaints = st.number_input("Number of Complaints", min_value=0, max_value=10, value=defaults["Number_of_Complaints"])
        
        avg_balance = st.number_input("Average Monthly Balance (MWK)", min_value=0, max_value=10000000, value=defaults["Average_Monthly_Balance"])
        avg_deposit = st.number_input("Average Monthly Deposit (MWK)", min_value=0, max_value=5000000, value=defaults["Average_Monthly_Deposit"])
        avg_withdrawal = st.number_input("Average Monthly Withdrawal (MWK)", min_value=0, max_value=5000000, value=defaults["Average_Monthly_Withdrawal"])
        txn_value_6mo = st.number_input("Total Transaction Value (6 months, MWK)", min_value=0, max_value=50000000, value=defaults["Total_Transaction_Value_Last_6_Months"])
        
        has_loan = st.checkbox("Has Active Loan", value=False)
        if has_loan:
            loan_status = st.selectbox("Loan Status", ["Active", "Active - Good Standing", "Active - Past Due"], index=1)
        else:
            loan_status = "No Active Loan"
        
        service_interactions = st.number_input("Customer Service Interactions", min_value=0, max_value=20, value=defaults["Customer_Service_Interactions"])
    
    # Predict button
    if st.button("Predict Churn Risk", type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Marital_Status": marital_status,
            "Education_Level": education,
            "Employment_Status": employment,
            "Monthly_Income": income,
            "Account_Tenure_Months": tenure,
            "Customer_Segment": segment,
            "Branch_Location": branch,
            "Number_of_Products": products,
            "Has_Credit_Card": int(has_credit),
            "Has_Mobile_Banking": int(has_mobile),
            "Has_Internet_Banking": int(has_internet),
            "Average_Monthly_Balance": avg_balance,
            "Average_Monthly_Transactions": avg_transactions,
            "Total_Transaction_Value_Last_6_Months": txn_value_6mo,
            "Number_of_Complaints": complaints,
            "Days_Since_Last_Transaction": days_inactive,
            "Average_Monthly_Deposit": avg_deposit,
            "Average_Monthly_Withdrawal": avg_withdrawal,
            "Has_Active_Loan": int(has_loan),
            "Loan_Status": loan_status,
            "Digital_Engagement_Score": digital_score,
            "Customer_Service_Interactions": service_interactions,
        }])
        
        # Make prediction
        proba = model.predict_proba(input_data)[0, 1]
        prediction = int(proba >= 0.50)
        
        # Get risk band
        recommender = RetentionRecommender()
        risk_band = recommender.assign_risk_band(proba)
        
        # Get explanation
        explanation = explain_prediction(model.pipeline, input_data, list(input_data.columns))
        
        # Display results
        st.markdown("---")
        
        # Risk band display
        if risk_band == "High Risk":
            risk_class = "risk-high"
        elif risk_band == "Medium Risk":
            risk_class = "risk-medium"
        else:
            risk_class = "risk-low"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>Churn Probability</h3><h1>{proba:.1%}</h1></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>Prediction</h3><h1>{"Churn" if prediction else "Retained"}</h1></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>Risk Band</h3><h1 class="{risk_class}">{risk_band}</h1></div>', unsafe_allow_html=True)
        
        # Explanation
        st.markdown('<h2 class="sub-header">Risk Factors</h2>', unsafe_allow_html=True)
        if explanation['reasons']:
            for reason in explanation['reasons']:
                st.markdown(f"- {reason}")
        else:
            st.info("No specific risk factors identified based on current profile.")
        
        # Retention recommendation
        st.markdown('<h2 class="sub-header">Retention Recommendation</h2>', unsafe_allow_html=True)
        rec = recommender.recommend_action(input_data.iloc[0], proba, risk_band)
        st.markdown(f"**Action:** {rec['recommended_action']}")
        st.markdown(f"**Priority:** {rec['priority']}")


def batch_prediction():
    """Batch prediction interface."""
    st.markdown('<h1 class="main-header">Batch Prediction</h1>', unsafe_allow_html=True)
    
    model = load_model()
    if model is None:
        return
    
    st.markdown("Upload a CSV file with customer data. The file should contain the same columns as the training data.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} customers")
            
            # Display sample
            with st.expander("Preview Data"):
                st.dataframe(df.head())
            
            # Predict
            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Running predictions..."):
                    proba = model.predict_proba(df)[:, 1]
                    predictions = (proba >= 0.50).astype(int)
                    
                    # Add predictions to dataframe
                    df['Churn_Probability'] = proba
                    df['Predicted_Churn'] = predictions
                    
                    # Add risk bands
                    recommender = RetentionRecommender()
                    df['Risk_Band'] = df['Churn_Probability'].apply(recommender.assign_risk_band)
                    
                    # Display results
                    st.success("Predictions complete!")
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("High Risk", (df['Risk_Band'] == 'High Risk').sum())
                    with col2:
                        st.metric("Medium Risk", (df['Risk_Band'] == 'Medium Risk').sum())
                    with col3:
                        st.metric("Low Risk", (df['Risk_Band'] == 'Low Risk').sum())
                    
                    # Results table
                    st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
                    result_cols = ['Customer_ID', 'Churn_Probability', 'Predicted_Churn', 'Risk_Band']
                    if 'Customer_ID' in df.columns:
                        st.dataframe(df[result_cols])
                    else:
                        st.dataframe(df[['Churn_Probability', 'Predicted_Churn', 'Risk_Band']])
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="churn_predictions_batch.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def model_info():
    """Model information page."""
    st.markdown('<h1 class="main-header">Model Information</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Model Details
    
    This application uses a Logistic Regression model trained on synthetic banking customer data.
    
    ### Model Performance (Test Set)
    - **ROC-AUC:** 0.74
    - **Accuracy:** 67.7%
    - **Recall:** 65.3%
    - **Precision:** 22.7%
    - **F1 Score:** 33.7%
    
    ### Comparison to Baseline
    The baseline model (always predicts "no churn") achieves:
    - Accuracy: 87.5% (by always predicting the majority class)
    - Recall: 0% (never identifies churned customers)
    
    Our model achieves 65.3% recall, meaning it can identify 65% of customers who will churn.
    
    ### Key Predictors (Permutation Importance)
    1. Days Since Last Transaction: 7.4%
    2. Account Tenure (months): 5.3%
    3. Number of Complaints: 4.1%
    4. Number of Products: 2.9%
    5. Average Monthly Deposit: 1.0%
    
    ### Risk Bands
    - **Low Risk:** Probability < 33% (3.9% observed churn rate)
    - **Medium Risk:** 33% ≤ Probability < 66% (12.1% observed churn rate)
    - **High Risk:** Probability ≥ 66% (33.1% observed churn rate)
    
    ### Important Notes
    - This model uses synthetic data for educational purposes.
    - Predictions are not calibrated for real-world banking use.
    - Feature importance shows association, not causation.
    - Do not use this model for actual customer decisions without proper validation.
    """)


def main():
    """Main application."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Single Prediction", "Batch Prediction", "Model Info"])
    
    if page == "Single Prediction":
        single_customer_prediction()
    elif page == "Batch Prediction":
        batch_prediction()
    else:
        model_info()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p><strong>Disclaimer:</strong> This is an educational portfolio project using synthetic data. 
    It is not intended for real banking decisions. Nyanja Commercial Bank is a fictional institution.</p>
    <p>Built with Streamlit • Logistic Regression Model • Synthetic Data</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
