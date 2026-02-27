import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    roc_auc_score
)

# -------------------- PATH SETUP --------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.predict import predict_churn
from src.feature_engineering import create_features

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Customer Churn Executive Dashboard",
    layout="wide"
)

# -------------------- STYLING --------------------
st.markdown("""
<style>
.big-title { font-size:40px !important; font-weight:700; }
.metric-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 10px;
    text-align:center;
    border: 1px solid #1f2937;
}
.risk-low { color: #22c55e; font-weight: bold; }
.risk-medium { color: #f59e0b; font-weight: bold; }
.risk-high { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">Telecom Customer Churn Executive Dashboard</p>', unsafe_allow_html=True)
st.caption("End-to-End ML Deployment | Business Intelligence | Executive Insights")
st.divider()

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction",
    "📊 Model Performance",
    "📈 Data Insights",
    "💼 Business Analytics"
])

# =========================
# TAB 1 - PREDICTION
# =========================
with tab1:

    st.sidebar.header("Customer Information")

    age = st.sidebar.slider("Age", 18, 80, 30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    region_circle = st.sidebar.selectbox("Region", ["West", "South", "North", "East", "Metro"])
    contract_type = st.sidebar.selectbox("Contract Type", ["No Contract", "Month-to-Month", "1 Year", "2 Year"])
    monthly_charges = st.sidebar.number_input("Monthly Charges", 100.0, 2000.0, 500.0)
    tenure_months = st.sidebar.slider("Tenure (Months)", 0, 120, 12)
    arpu = st.sidebar.number_input("ARPU", 100.0, 3000.0, 500.0)
    nps_score = st.sidebar.slider("NPS Score", -100, 100, 10)
    num_complaints_3m = st.sidebar.slider("Complaints (3M)", 0, 10, 1)

    if st.sidebar.button("Predict Churn Risk"):

        input_data = {
            "age": age,
            "gender": gender,
            "region_circle": region_circle,
            "contract_type": contract_type,
            "monthly_charges": monthly_charges,
            "tenure_months": tenure_months,
            "arpu": arpu,
            "nps_score": nps_score,
            "num_complaints_3m": num_complaints_3m
        }

        result = predict_churn(input_data)
        prob = result["churn_probability"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Churn Probability", f"{prob:.2%}")

        with col2:
            if prob < 0.30:
                st.markdown('<p class="risk-low">LOW RISK</p>', unsafe_allow_html=True)
            elif prob < 0.60:
                st.markdown('<p class="risk-medium">MEDIUM RISK</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="risk-high">HIGH RISK</p>', unsafe_allow_html=True)

        st.progress(min(prob, 1.0))

# =========================
# LOAD DATA ONCE
# =========================
df = pd.read_csv("data/raw/telecom_churn.csv")
df = df.drop(columns=["customer_id"])
df = create_features(df)

X = df.drop(columns=["is_churn"])
y = df["is_churn"]

model = joblib.load("models/churn_model.pkl")
y_probs = model.predict_proba(X)[:, 1]
y_pred = (y_probs >= 0.40).astype(int)

# =========================
# TAB 2 - MODEL PERFORMANCE
# =========================
with tab2:

    st.subheader("Model Performance Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        accuracy = accuracy_score(y, y_pred)
        roc_score = roc_auc_score(y, y_probs)
        st.metric("Accuracy", f"{accuracy:.2f}")
        st.metric("ROC-AUC", f"{roc_score:.2f}")

    with col2:
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
        st.pyplot(fig)

    st.divider()

    fpr, tpr, _ = roc_curve(y, y_probs)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0,1], [0,1], linestyle="--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    st.pyplot(fig2)

    precision, recall, _ = precision_recall_curve(y, y_probs)

    fig3, ax3 = plt.subplots()
    ax3.plot(recall, precision)
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    st.pyplot(fig3)

# =========================
# TAB 3 - DATA INSIGHTS
# =========================
with tab3:

    st.subheader("Customer Churn Insights")

    churn_rate = df["is_churn"].mean()
    st.metric("Overall Churn Rate", f"{churn_rate:.2%}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Churn by Region")
        region_churn = df.groupby("region_circle")["is_churn"].mean()
        st.bar_chart(region_churn)

    with col2:
        st.markdown("### Churn by Contract Type")
        contract_churn = df.groupby("contract_type")["is_churn"].mean()
        st.bar_chart(contract_churn)

    st.markdown("### Churn Distribution")
    st.bar_chart(df["is_churn"].value_counts())

# =========================
# TAB 4 - BUSINESS ANALYTICS
# =========================
with tab4:

    st.subheader("Executive Business Metrics")

    high_value = df[df["arpu"] > 500]
    high_value_churn = high_value["is_churn"].mean()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("High Value Churn Rate", f"{high_value_churn:.2%}")

    with col2:
        churned = df[df["is_churn"] == 1]
        revenue_at_risk = churned["arpu"].sum()
        st.metric("Revenue at Risk", f"₹{revenue_at_risk:,.0f}")

    with col3:
        region_churn = df.groupby("region_circle")["is_churn"].mean()
        highest_region = region_churn.idxmax()
        st.metric("Highest Risk Region", highest_region)