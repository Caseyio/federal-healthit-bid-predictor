import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Federal Health IT Award Estimator", layout="centered")

# --- Load model and feature columns ---
model = joblib.load("xgb_federal_award_model.pkl")
feature_cols = joblib.load("xgb_feature_columns.pkl")

st.title("Federal Health IT Bid Amount Predictor")
st.write("Estimate the award amount for a federal Health IT contract based on contract features.")

# --- User Inputs ---
st.header("Enter Contract Details")

naics_code = st.selectbox("NAICS Code", ["541511", "541512", "541519"])
pricing_type = st.selectbox("Pricing Type", [
    "FIRM FIXED PRICE",
    "COST PLUS AWARD FEE",
    "COST PLUS FIXED FEE"
])
offers_received = st.slider("Number of Offers Received", min_value=1, max_value=10, value=3)
set_aside = st.selectbox("Set-Aside Type", [
    "NO SET ASIDE USED.",
    "SMALL BUSINESS SET ASIDE - TOTAL"
])
is_va = st.checkbox("Department of Veterans Affairs is the awarding agency")

# Build input with all expected columns defaulted to 0
input_data = {col: 0 for col in feature_cols}

# Populate input_data from user selections
input_data[f"naics_code_{naics_code}"] = 1
input_data[f"pricing_type_{pricing_type}"] = 1
input_data["offers_received"] = offers_received
input_data[f"set_aside_{set_aside}"] = 1
if is_va:
    input_data["agency_Department of Veterans Affairs"] = 1

# Convert to DataFrame and enforce column order
input_df = pd.DataFrame([input_data])[feature_cols]


# --- Predict and Show Result ---
if st.button("Predict Award Amount"):
    predicted_log = model.predict(input_df)
    predicted_amount = np.expm1(predicted_log[0])
    st.success(f"Estimated Award Amount: ${predicted_amount:,.2f}")
    
st.caption("This model is trained on over 43,000 real Health IT federal contracts between 2018–2025.")
