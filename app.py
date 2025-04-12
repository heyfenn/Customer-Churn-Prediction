import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.markdown("""
# 📉 Customer Churn Prediction App
Predict whether a telecom customer is likely to churn based on their service, contract, and payment details.
""")

# Load model and preprocessing assets
@st.cache_resource
def load_assets():
    with open("tuned_rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, label_encoders, scaler, feature_names

model, label_encoders, scaler, feature_names = load_assets()

# Define feature categories
user_data_cols = ["Gender", "Senior Citizen", "Partner", "Dependents"]
payment_data_cols = ["Monthly Charges", "Total Charges", "Payment Method", "Paperless Billing"]
contract_data_cols = ["Contract", "Tenure Months"]
services_data_cols = ["Phone Service", "Multiple Lines", "Internet Service", "Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"]

derived_features = ["Avg_Monthly_Charges", "Tenure_Group"]

input_data = {}

# Expandable sections for all user input
with st.expander("👤 User Profile"):
    for col in user_data_cols:
        if col in label_encoders:
            input_data[col] = st.selectbox(col, label_encoders[col].classes_.tolist())

with st.expander("💳 Payment Info"):
    for col in payment_data_cols:
        if col in label_encoders:
            input_data[col] = st.selectbox(col, label_encoders[col].classes_.tolist())
        else:
            input_data[col] = st.number_input(col, value=0.0, step=0.1)

with st.expander("📄 Contract & Tenure"):
    for col in contract_data_cols:
        if col == "Tenure Months":
            input_data[col] = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        elif col in label_encoders:
            input_data[col] = st.selectbox(col, label_encoders[col].classes_.tolist())

with st.expander("📶 Services Subscribed"):
    for col in services_data_cols:
        if col in label_encoders:
            input_data[col] = st.selectbox(col, label_encoders[col].classes_.tolist())

# Prediction
st.markdown("---")
if st.button("🔮 Predict Churn"):
    st.markdown("## 📌 Customer Details Overview")
    input_df = pd.DataFrame([input_data])

    # Derived Features
    input_df["Avg_Monthly_Charges"] = input_df["Total Charges"] / input_df["Tenure Months"].replace(0, 1)
    tenure_max = max(input_df["Tenure Months"].values[0], 61)
    input_df["Tenure_Group"] = pd.cut(input_df["Tenure Months"],
                                       bins=[0, 12, 24, 48, 60, tenure_max],
                                       labels=['0-12', '13-24', '25-48', '49-60', '60+']).astype(str)

    # Encode categoricals
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # Fill in any missing columns expected by model
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    st.dataframe(input_df.style.format(precision=2), use_container_width=True)

    # Make prediction
    scaled_input = scaler.transform(input_df)
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1] * 100

    st.markdown("## 🔍 Prediction Result")
    if pred == 1:
        st.error("🚨 The customer is likely to churn")
    else:
        st.success("✅ The customer is likely to stay")
    st.metric(label="Churn Probability", value=f"{prob:.2f}%")
