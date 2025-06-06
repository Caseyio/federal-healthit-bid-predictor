import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load model and feature columns ---
model = joblib.load("xgb_federal_award_model.pkl")
feature_cols = joblib.load("xgb_feature_columns.pkl")

# Define RMSE from training to use for confidence range
RMSE_LOG = 1.71
MARGIN = np.expm1(RMSE_LOG)  # ~4.5x error margin on actual award dollars

st.title("Federal Health IT Bid Confidence Tool")
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

# --- Build Input Row with Full Feature Set ---
input_data = {col: 0 for col in feature_cols}
input_data[f"naics_code_{naics_code}"] = 1
input_data[f"pricing_type_{pricing_type}"] = 1
input_data["offers_received"] = offers_received
input_data[f"set_aside_{set_aside}"] = 1
if is_va:
    input_data["agency_Department of Veterans Affairs"] = 1

# Ensure input_df has same structure and order as training data
input_df = pd.DataFrame([input_data])[feature_cols]

# --- Predict and Show Confidence Range ---
# --- Predict and Show Confidence Range ---
if st.button("Calculate Confidence Range"):
    predicted_log = model.predict(input_df)
    predicted_amount = np.expm1(predicted_log[0])
    lower = predicted_amount / MARGIN
    upper = predicted_amount * MARGIN

    # Exact Streamlit success-style green boxes
    box_style = """
        background-color: #dff0d8;
        padding: 1rem;
        border-radius: 4px;
        border: 1px solid #d0e9c6;
        color: #3c763d;
        font-size: 16px;
        margin-bottom: 0.75rem;
    """

    st.markdown(f"""
    <div style="{box_style}">
        <strong>Estimated Award Amount:</strong> ${predicted_amount:,.2f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="{box_style}">
        <strong>Confidence Range (+/- 1.71 RMSE):</strong> ${lower:,.2f} – ${upper:,.2f}
    </div>
    """, unsafe_allow_html=True)

    st.caption("This model is trained on over 43,000 real Health IT federal contracts between 2018–2025.")
