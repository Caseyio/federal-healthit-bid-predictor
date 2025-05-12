# streamlit_app.py
# ---------------------------------------------------
# Streamlit app to predict federal contract award amounts
# Author: Casey Ortiz | kcarlos.ortiz@gmail.com
# ---------------------------------------------------

# Optional Streamlit config block
# --------------------------------
# streamlit config YAML block (optional in .streamlit/config.toml)
# --------------------------------
# Streamlit will ignore this if not in a config file

# NOTE: If you want to set title/favicon officially,
# put this in a `.streamlit/config.toml` instead

# Example config.toml (optional, see below):
# [theme]
# primaryColor = "#1E88E5"
# backgroundColor = "#F5F9FF"
# secondaryBackgroundColor = "#E8F0FE"
# textColor = "#262730"
# font = "sans serif"

import streamlit as st
import pandas as pd
import numpy as np
import joblib

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

# --- Build Input Row ---
input_data = {col: 0 for col in feature_cols}

# Map inputs to the right one-hot columns
input_data[f"naics_code_{naics_code}"] = 1
input_data[f"pricing_type_{pricing_type}"] = 1
input_data["offers_received"] = offers_received
input_data[f"set_aside_{set_aside}"] = 1
if is_va:
    input_data["agency_Department of Veterans Affairs"] = 1

input_df = pd.DataFrame([input_data])

# --- Predict and Show Result ---
if st.button("Predict Award Amount"):
    predicted_log = model.predict(input_df)
    predicted_amount = np.expm1(predicted_log[0])
    st.success(f"Estimated Award Amount: ${predicted_amount:,.2f}")
