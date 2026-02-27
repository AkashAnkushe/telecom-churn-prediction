import streamlit as st
import sys
import os
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:700;
}

.metric-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 10px;
    text-align:center;
    border: 1px solid #1f2937;
}

.risk-low {
    color: #22c55e;
    font-weight: bold;
}

.risk-medium {
    color: #f59e0b;
    font-weight: bold;
}

.risk-high {
    color: #ef4444;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.predict import predict_churn

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    layout="wide"
)

st.markdown('<p class="big-title">Customer Churn Prediction Dashboard</p>', unsafe_allow_html=True)
st.caption("Advanced ML-Based Churn Risk Detection System")
st.divider()
st.markdown("Advanced ML-based Churn Risk Detection System")

st.sidebar.header("Customer Information")

# --- Sidebar Inputs ---
age = st.sidebar.slider("Age", 18, 80, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
region_circle = st.sidebar.selectbox("Region", ["West", "South", "North", "East", "Metro"])
connection_type = st.sidebar.selectbox("Connection Type", ["4G", "5G", "Fiber Home Broadband"])
plan_type = st.sidebar.selectbox("Plan Type", ["Prepaid", "Postpaid"])
contract_type = st.sidebar.selectbox("Contract Type", ["No Contract", "Month-to-Month", "1 Year", "2 Year"])
base_plan_category = st.sidebar.selectbox(
    "Base Plan Category",
    ["Postpaid Platinum", "Prepaid Unlimited", "Prepaid Regular",
     "Prepaid Mini", "Postpaid Silver", "Postpaid Gold"]
)
segment_value = st.sidebar.selectbox("Customer Segment", ["Low", "Medium", "High"])

tenure_months = st.sidebar.slider("Tenure (Months)", 0, 120, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 100.0, 2000.0, 500.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 200000.0, 6000.0)
avg_data_gb_month = st.sidebar.slider("Avg Data Usage (GB)", 0, 100, 20)
avg_voice_mins_month = st.sidebar.slider("Avg Voice Minutes", 0, 2000, 500)
sms_count_month = st.sidebar.slider("SMS Count", 0, 200, 50)
overage_charges = st.sidebar.number_input("Overage Charges", 0.0, 1000.0, 50.0)

is_family_plan = st.sidebar.selectbox("Family Plan", [0, 1])
is_multi_service = st.sidebar.selectbox("Multi Service", [0, 1])
network_issues_3m = st.sidebar.slider("Network Issues (3M)", 0, 10, 1)
dropped_call_rate = st.sidebar.slider("Dropped Call Rate", 0.0, 1.0, 0.02)
avg_data_speed_mbps = st.sidebar.slider("Avg Data Speed (Mbps)", 1, 100, 20)

num_complaints_3m = st.sidebar.slider("Complaints (3M)", 0, 10, 1)
num_complaints_12m = st.sidebar.slider("Complaints (12M)", 0, 20, 2)
call_center_interactions_3m = st.sidebar.slider("Call Center Interactions", 0, 10, 1)
last_complaint_resolution_days = st.sidebar.slider("Complaint Resolution Days", 0, 30, 3)

app_logins_30d = st.sidebar.slider("App Logins (30D)", 0, 30, 5)
selfcare_transactions_30d = st.sidebar.slider("Selfcare Transactions", 0, 20, 2)
auto_pay_enrolled = st.sidebar.selectbox("Auto Pay Enrolled", [0, 1])
late_payment_flag_3m = st.sidebar.selectbox("Late Payment (3M)", [0, 1])
avg_payment_delay_days = st.sidebar.slider("Avg Payment Delay Days", 0, 15, 2)

arpu = st.sidebar.number_input("ARPU", 100.0, 3000.0, 500.0)
nps_score = st.sidebar.slider("NPS Score", -100, 100, 10)
service_rating_last_6m = st.sidebar.slider("Service Rating (Last 6M)", 1, 5, 3)

received_competitor_offer_flag = st.sidebar.selectbox("Received Competitor Offer", [0, 1])
retention_offer_accepted_flag = st.sidebar.selectbox("Retention Offer Accepted", [0, 1])

# --- Predict Button ---
if st.sidebar.button("Predict Churn Risk"):

    input_data = {
        "age": age,
        "gender": gender,
        "region_circle": region_circle,
        "connection_type": connection_type,
        "plan_type": plan_type,
        "contract_type": contract_type,
        "base_plan_category": base_plan_category,
        "segment_value": segment_value,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "avg_data_gb_month": avg_data_gb_month,
        "avg_voice_mins_month": avg_voice_mins_month,
        "sms_count_month": sms_count_month,
        "overage_charges": overage_charges,
        "is_family_plan": is_family_plan,
        "is_multi_service": is_multi_service,
        "network_issues_3m": network_issues_3m,
        "dropped_call_rate": dropped_call_rate,
        "avg_data_speed_mbps": avg_data_speed_mbps,
        "num_complaints_3m": num_complaints_3m,
        "num_complaints_12m": num_complaints_12m,
        "call_center_interactions_3m": call_center_interactions_3m,
        "last_complaint_resolution_days": last_complaint_resolution_days,
        "app_logins_30d": app_logins_30d,
        "selfcare_transactions_30d": selfcare_transactions_30d,
        "auto_pay_enrolled": auto_pay_enrolled,
        "late_payment_flag_3m": late_payment_flag_3m,
        "avg_payment_delay_days": avg_payment_delay_days,
        "arpu": arpu,
        "nps_score": nps_score,
        "service_rating_last_6m": service_rating_last_6m,
        "received_competitor_offer_flag": received_competitor_offer_flag,
        "retention_offer_accepted_flag": retention_offer_accepted_flag
    }

    result = predict_churn(input_data)

    prob = result["churn_probability"]
    age_value = age
    arpu_value = arpu
    nps_value = nps_score
    complaints_value = num_complaints_3m
    prediction = result["churn_prediction"]

    st.markdown("## Prediction Result")

    st.markdown(f"""
    <div class="metric-box">
        <h2 style="margin:0;">Churn Probability</h2>
        <h1 style="margin:0; font-size:48px;">{prob:.2%}</h1>
    </div>
    """, unsafe_allow_html=True)

    # Risk Level Logic
    if prob < 0.30:
        st.markdown('<p class="risk-low">🟢 LOW RISK OF CHURN</p>', unsafe_allow_html=True)
        recommendation = "Maintain engagement through loyalty rewards and cross-sell opportunities."

    elif 0.30 <= prob < 0.60:
        st.markdown('<p class="risk-medium">🟡 MEDIUM RISK OF CHURN</p>', unsafe_allow_html=True)
        recommendation = "Offer targeted discount or personalized retention call."

    else:
        st.markdown('<p class="risk-high">🔴 HIGH RISK OF CHURN</p>', unsafe_allow_html=True)

        # Smart recommendation logic
        if nps_value < 20:
            recommendation = "Customer satisfaction is low. Prioritize service improvement call and issue resolution."

        elif complaints_value >= 2:
            recommendation = "High complaints detected. Assign senior support team and fast-track resolution."

        elif arpu_value > 500:
            recommendation = "High value customer. Provide premium retention offer and loyalty benefits."

        elif age_value < 30:
            recommendation = "Young segment. Offer OTT bundle, data upgrade, or digital perks."

        else:
            recommendation = "Provide personalized retention package with flexible contract options."

    st.info(f"📌 Recommendation: {recommendation}")
    st.progress(min(prob, 1.0))
    
    
import pandas as pd
import numpy as np
import joblib

st.divider()
st.markdown("## 🔎 Top Churn Drivers")

# Load model
model = joblib.load("models/churn_model.pkl")

# Extract coefficients
coefficients = model.named_steps["classifier"].coef_[0]
feature_names = model.named_steps["preprocessing"].get_feature_names_out()

# Create dataframe
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

importance_df["Abs_Coefficient"] = np.abs(importance_df["Coefficient"])

# Get top 10 important features
top_features = importance_df.sort_values(
    by="Abs_Coefficient",
    ascending=False
).head(10)

# Clean feature names (optional)
top_features["Feature"] = top_features["Feature"].str.replace("num__", "", regex=False)
top_features["Feature"] = top_features["Feature"].str.replace("cat__", "", regex=False)

st.bar_chart(top_features.set_index("Feature")["Abs_Coefficient"])

st.divider()

with st.expander("📦 Model Details (Technical Overview)"):

    st.write("**Model Type:** Logistic Regression")
    st.write("**ROC-AUC:** 0.65")
    st.write("**Accuracy:** 0.54 (threshold tuned for higher churn recall)")
    st.write("**Threshold Used:** 0.40")
    st.write("**Training Samples:** 20,000")
    st.write("**Test Samples:** 5,000")

    st.info("Model uses feature engineering + segmentation-aware threshold tuning for churn prediction.")