# app.py
import streamlit as st
import numpy as np
import pandas as pd
from utils import load_model, preprocess_input, predict_default

# Load model
model, scaler = load_model()

# App title
st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title("🏦 Loan Default Prediction System")
st.markdown("Predict the likelihood of a customer defaulting on a loan using machine learning.")

# Sidebar inputs
st.sidebar.header("🧾 Applicant Details")

Duration_in_month = st.sidebar.number_input("Loan Duration (months)", 6, 72, 24)
Credit_amount = st.sidebar.number_input("Loan Amount", 250, 20000, 2000)
Installment_rate_in_percentage_of_disposable_income = st.sidebar.slider("Installment Rate (%)", 1, 4, 2)
Age_in_years = st.sidebar.number_input("Age (years)", 18, 75, 35)
Number_of_existing_credits_at_this_bank = st.sidebar.slider("Existing Credits", 1, 4, 1)

# Optional advanced inputs
st.sidebar.markdown("### Other Financial Info")
Duration_in_month = float(Duration_in_month)
Credit_amount = float(Credit_amount)
Age_in_years = float(Age_in_years)

# Collect features into dictionary (must match training order)
input_data = {
    "Duration_in_month": Duration_in_month,
    "Credit_amount": Credit_amount,
    "Installment_rate_in_percentage_of_disposable_income": Installment_rate_in_percentage_of_disposable_income,
    "Age_in_years": Age_in_years,
    "Number_of_existing_credits_at_this_bank": Number_of_existing_credits_at_this_bank
}

st.write("### Input Summary:")
st.write(pd.DataFrame([input_data]))

# Predict button
if st.button("🔍 Predict Loan Default Risk"):
    scaled_input = preprocess_input(input_data, scaler)
    pred, prob = predict_default(model, scaled_input)

    if pred == 1:
        st.error(f"⚠️ High Risk: Likely to Default (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk: Likely to Repay (Probability: {prob:.2f})")

